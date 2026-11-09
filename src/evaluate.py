

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = next(
    (
        parent / "participant"
        for parent in ROOT.parents
        if (parent / "participant").is_dir()
    ),
    ROOT.parent / "participant",
)


def parse(value):
    """Decode an absent marker or an (x, y, membership) CSV cell."""
    if value.strip() == "-1":
        return None
    values = ast.literal_eval(value)
    if len(values) not in (2, 3):
        raise ValueError(f"Invalid prediction: {value}")
    return np.array(
        [values[0], values[1], values[2] if len(values) == 3 else 0], dtype=float
    )


def reward(distance):
    """Give full credit within 12 pixels, declining linearly to zero at 36 pixels."""
    return np.clip((36 - np.asarray(distance)) / 24, 0, 1)


def score_scene(truth, prediction):
    """Compute the four competition metrics for one labelled scene."""
    count = int(truth["n_patches"])
    if int(prediction["n_patches"]) != count:
        raise ValueError("n_patches differs from the reference CSV")
    actual = [parse(truth[f"patch_{i:02}"]) for i in range(1, count + 1)]
    guesses = [parse(prediction[f"patch_{i:02}"]) for i in range(1, count + 1)]
    true_presence = np.array([point is not None for point in actual])
    pred_presence = np.array([point is not None for point in guesses])
    f1 = []

    # Treat present and absent as separate positive classes, then average their F1
    # scores.
    for positive in [False, True]:
        tp = np.sum((true_presence == positive) & (pred_presence == positive))
        fp = np.sum((true_presence != positive) & (pred_presence == positive))
        fn = np.sum((true_presence == positive) & (pred_presence != positive))
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    localization = [
        float(reward(np.linalg.norm(a[:2] - b[:2]))) if b is not None else 0
        for a, b in zip(actual, guesses)
        if a is not None
    ]

    # Geometric recovery uses all reported centres, irrespective of predicted membership
    # bits.
    figure = np.array(
        [point[:2] for point in actual if point is not None and point[2] == 1]
    )
    reported = np.array([point[:2] for point in guesses if point is not None])
    geometric = 0.0
    if len(figure) and len(reported):
        distances = np.linalg.norm(figure[:, None] - reported[None], axis=2)
        used_truth, used_prediction = (set(), set())

        # Greedy nearest-pair matching follows the competition metric, not the inference
        # solver.
        for flat in np.argsort(distances, axis=None):
            a, b = np.unravel_index(flat, distances.shape)
            if a not in used_truth and b not in used_prediction:
                geometric += float(reward(distances[a, b]))
                used_truth.add(a)
                used_prediction.add(b)
        geometric /= len(figure)
    result = dict(
        presence=float(np.mean(f1)),
        localization=float(np.mean(localization)) if localization else 0.0,
        geometric_recovery=geometric,
        identification=float(truth["constellation"] == prediction["constellation"]),
    )
    result["score"] = sum(
        (result[key] * weight for key, weight in zip(result, [0.25, 0.2, 0.25, 0.3]))
    )
    return result


def evaluate(truth_path, predictions_path):
    """Validate row IDs and average each metric equally across scenes."""
    with truth_path.open(newline="", encoding="utf-8-sig") as file:
        truth = list(csv.DictReader(file))
    with predictions_path.open(newline="", encoding="utf-8-sig") as file:
        predictions = list(csv.DictReader(file))
    indexed = {row["Id"]: row for row in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {row["Id"] for row in truth}:
        raise ValueError("Predictions must contain every reference scene exactly once")
    scores = {row["Id"]: score_scene(row, indexed[row["Id"]]) for row in truth}
    scores["mean"] = {
        key: float(np.mean([scene[key] for scene in scores.values()]))
        for key in next(iter(scores.values()))
    }
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth", type=Path, default=DEFAULT_DATA / "train_ground_truth.csv"
    )
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "results" / "train_predictions.csv"
    )
    args = parser.parse_args()
    results = evaluate(args.truth, args.predictions)
    print(json.dumps(results, indent=2))
    args.predictions.with_suffix(".metrics.json").write_text(
        json.dumps(results, indent=2)
    )
