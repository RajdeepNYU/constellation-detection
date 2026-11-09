

import argparse
import csv
import hashlib
import json
import os
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from matching import PatchMatcher
from patterns import PatternMatcher, bright_stars
from assignment import assign_figure

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = next(
    (
        parent / "participant"
        for parent in ROOT.parents
        if (parent / "participant").is_dir()
    ),
    ROOT.parent / "participant",
)


def read_image(path):
    """Read a grayscale image and fail early if the path cannot be decoded."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def fingerprint(folder, scene_id, count, fast=False):
    """Identify a localization cache by matcher source, mode and input bytes.

    Even comment changes in matching.py invalidate old caches deliberately.
    Geometry is recomputed on every run; only patch candidates are cached.
    """
    digest = hashlib.sha256((ROOT / "matching.py").read_bytes())
    digest.update(b"fast" if fast else b"full")
    files = [folder / f"{scene_id}_image.png"]
    files += [folder / "patches" / f"patch_{i:02}.png" for i in range(1, count + 1)]
    for file in files:
        digest.update(file.read_bytes())
    return digest.hexdigest()


def merge_candidates(first, second):
    """Merge two score-sorted candidate lists, keeping centres over six pixels apart."""
    merged = []
    for candidate in sorted(first + second, key=lambda c: c["score"], reverse=True):
        if all(
            (
                np.linalg.norm(np.array(candidate["xy"]) - previous["xy"]) > 6
                for previous in merged
            )
        ):
            merged.append(candidate)
    return merged


def process_scene(
    data, split, row, cache_dir, resume, threshold, margin, fast=False, balanced=False
):
    """Locate queries, fit a constellation and produce one submission row.

    Candidates map patch column names to dictionaries containing xy, score
    and appearance. Model dictionaries hold projected stars and fit scores.
    Coordinates remain in original-image pixels after the coarse search.
    """
    started = perf_counter()
    cv2.setRNGSeed(42)
    scene_id, count = (row["Id"], int(row["n_patches"]))
    folder = data / split / scene_id
    print(f"{scene_id}: locating {count} patches", flush=True)
    image = read_image(folder / f"{scene_id}_image.png")
    matcher = PatchMatcher(image, fast=fast)
    cache_path = cache_dir / ("fast" if fast else "full") / split / f"{scene_id}.json"

    # Reuse localization only when the matcher and every input file are unchanged.
    signature = fingerprint(folder, scene_id, count, fast)
    cached = (
        json.loads(cache_path.read_text()) if resume and cache_path.exists() else {}
    )
    if cached.get("fingerprint") == signature:
        candidates = cached["candidates"]
        print(f"{scene_id}: reused verified localization cache", flush=True)
    else:
        candidates = {}
        for index in range(1, count + 1):
            column = f"patch_{index:02}"
            candidates[column] = matcher.locate(
                read_image(folder / "patches" / f"{column}.png")
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(dict(fingerprint=signature, candidates=candidates))
        )
    located = perf_counter()
    print(f"{scene_id}: aligning reference constellations", flush=True)

    # Fit reference geometry, then use its nodes to guide additional patch checks.
    models = PatternMatcher(data / "patterns", fast=fast).propose(candidates, image)
    stars, centers = (
        bright_stars(
            image,
            adaptive=any((model.get("adaptive_stars", False) for model in models)),
        ),
        [],
    )
    for model in models:
        projected = np.array(model["projected"])
        distances = np.linalg.norm(projected[:, None] - stars[None], axis=2)
        indices = distances.argmin(1)
        for point in stars[indices[distances.min(1) < 12]]:
            if all((np.linalg.norm(point - other) > 8 for other in centers)):
                centers.append(point)
    library = matcher.prepare_targets(centers)
    if library is not None:
        for column in candidates:
            patch = read_image(folder / "patches" / f"{column}.png")
            candidates[column] = merge_candidates(
                candidates[column], matcher.targeted(patch, library)
            )
    first_pass = perf_counter()
    choices = [(assign_figure(candidates, model), model) for model in models]
    choices.sort(key=lambda choice: choice[0][0], reverse=True)

    # In fast mode, revisit uncertain queries when two figures have similar support.
    if (
        fast
        and len(choices) > 1
        and (choices[0][0][0] - choices[1][0][0] < 0.12)
        and (library is not None)
    ):
        matcher.fast = False
        for column, items in candidates.items():
            gap = items[0]["score"] - items[1]["score"] if len(items) > 1 else 1.0
            if gap < 0.1:
                patch = read_image(folder / "patches" / f"{column}.png")
                candidates[column] = merge_candidates(
                    items, matcher.targeted(patch, library)
                )
        matcher.fast = True
        choices = [(assign_figure(candidates, model), model) for model in models]
        choices.sort(key=lambda choice: choice[0][0], reverse=True)
    baseline_score = choices[0][0][0] if choices else 0.0
    best_model = choices[0][1] if choices else None
    reliable = (
        best_model is not None
        and best_model["image_score"] >= 4.5
        and (best_model["matched_stars"] / best_model["total_stars"] >= 0.75)
    )

    # Expand the search only when the first figure lacks sufficient image support.
    if not reliable:
        print(f"{scene_id}: expanding weak geometry", flush=True)
        recovered = PatternMatcher(data / "patterns", fast=fast).recover(
            candidates, image, fast=fast, balanced=balanced
        )
        ranked = []
        for model in recovered:
            score, selected = assign_figure(candidates, model)

            # Four points can fit a homography by chance; require extra query support.
            score *= min(1.0, max(0, len(selected) - 4) / 3)
            model["recovery_score"] = score
            if score >= max(0.75, baseline_score * 1.15):
                ranked.append(model)
        ranked.sort(key=lambda model: model["recovery_score"], reverse=True)
        additions = []
        for model in ranked:
            if any(
                (
                    model["name"] == other["name"]
                    and np.mean(
                        np.linalg.norm(
                            np.array(model["projected"]) - other["projected"], axis=1
                        )
                    )
                    < 10
                    for other in additions
                )
            ):
                continue
            additions.append(model)
            if len(additions) == 3:
                break
        if additions:
            centers = []
            stars = bright_stars(image, adaptive=True)
            for model in additions:
                projected = np.array(model["projected"])
                for point in projected:
                    distance = np.linalg.norm(stars - point, axis=1)
                    target = stars[distance.argmin()] if distance.min() < 4 else point
                    if (
                        48 <= target[0] < image.shape[1] - 48
                        and 48 <= target[1] < image.shape[0] - 48
                        and all((np.linalg.norm(target - old) > 8 for old in centers))
                    ):
                        centers.append(target)
            library = matcher.prepare_targets(centers)
            if library is not None:
                for column in candidates:
                    patch = read_image(folder / "patches" / f"{column}.png")
                    candidates[column] = merge_candidates(
                        candidates[column], matcher.targeted(patch, library)
                    )
            models.extend(additions)
            choices = [(assign_figure(candidates, model), model) for model in models]
            choices.sort(key=lambda choice: choice[0][0], reverse=True)

    # Choose the best joint figure fit before deciding which remaining queries are
    # present.
    name, members = ("unknown", {})
    if choices:
        (joint_score, members), model = choices[0]
        name = model["name"]
    for (joint_score, _), model in choices:
        model["joint_score"] = joint_score
    output = {field: "-1" for field in row}
    output.update(Id=scene_id, n_patches=str(count), constellation=name)
    for column, items in candidates.items():
        candidate, member = (None, 0)
        if column in members:
            candidate, _ = members[column]
            member = 1

        # Off-figure matches need both strong appearance and a clear lead over
        # alternatives.
        elif items:
            best = items[0]
            gap = best["score"] - (items[1]["score"] if len(items) > 1 else 0)
            if (
                best["score"] >= threshold
                and best["appearance"] >= 0.65
                and (gap >= margin)
            ):
                candidate = best
        if candidate is not None:
            x, y = np.round(candidate["xy"]).astype(int)
            output[column] = f"({x}, {y}, {member})"
    present = sum((output[f"patch_{i:02}"] != "-1" for i in range(1, count + 1)))
    print(f"{scene_id}: {name}, {present}/{count} present", flush=True)
    return (
        output,
        dict(
            models=[model for _, model in choices],
            patches=candidates,
            timing=dict(
                localization=located - started,
                initial_geometry_and_recheck=first_pass - located,
                recovery_and_decisions=perf_counter() - first_pass,
                total=perf_counter() - started,
            ),
        ),
    )


def run_scene(job):
    """Run one worker job with OpenCV internal threading disabled."""
    cv2.setNumThreads(1)
    return process_scene(*job)


def run(args):
    """Process the selected split and save its CSV, diagnostics and wall-clock timing."""
    started = perf_counter()
    cv2.setNumThreads(1)
    data = args.data.resolve()
    template = data / (
        "train_ground_truth.csv" if args.split == "train" else "sample_submission.csv"
    )
    with template.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields, rows = (reader.fieldnames, list(reader))
    if len({row["Id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate scene IDs in input CSV")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output.parent / "cache"
    jobs = [
        (
            data,
            args.split,
            row,
            cache_dir,
            args.resume,
            args.threshold,
            args.margin,
            args.fast,
            args.balanced,
        )
        for row in rows
    ]
    predictions, diagnostics = ([], {})
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for prediction, details in executor.map(run_scene, jobs):
            predictions.append(prediction)
            diagnostics[prediction["Id"]] = details

    # Publish a CSV only after every scene succeeds, so partial runs cannot replace it.
    temporary = args.output.with_suffix(".partial.csv")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    temporary.replace(args.output)
    args.output.with_suffix(".diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2)
    )
    args.output.with_suffix(".timing.json").write_text(
        json.dumps(
            dict(
                seconds=perf_counter() - started,
                fast=args.fast,
                balanced=args.balanced,
                resume=args.resume,
                workers=args.workers,
                scenes={
                    name: details["timing"] for name, details in diagnostics.items()
                },
            ),
            indent=2,
        )
    )
    print(f"Saved {args.output} ({len(predictions)} scenes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--split", choices=["train", "validation"], default="validation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to train_predictions.csv for train, submission.csv for validation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Defaults to up to 6 threads in balanced mode, 3 threads otherwise",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse localization only when image, patches and matcher code are unchanged",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--balanced",
        action="store_true",
        help="Full-quality matching with parallel workers and selective recovery",
    )
    modes.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer image searches and skip expensive projective recovery",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.07)
    args = parser.parse_args()
    if args.output is None:
        stem = "train_predictions" if args.split == "train" else "submission"
        args.output = (
            ROOT
            / "results"
            / (
                stem
                + (
                    "_fast.csv"
                    if args.fast
                    else "_balanced.csv" if args.balanced else ".csv"
                )
            )
        )
    if args.workers is None:
        args.workers = (
            min(6, max(1, (os.cpu_count() or 2) // 2)) if args.balanced else 3
        )
    if args.workers < 1:
        parser.error("--workers must be positive")
    run(args)
