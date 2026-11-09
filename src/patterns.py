

from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


def read_stars(path):
    """Extract bright diagram nodes, then centre and normalize their coordinates."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    mask = np.all(image[:, :, :3] > 180, axis=2)
    if image.shape[2] == 4:
        mask &= image[:, :, 3] > 100
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype("uint8"))
    points = centers[1:][stats[1:, cv2.CC_STAT_AREA] >= 8]
    if len(points):
        points = (points - points.mean(0)) / max(np.ptp(points, axis=0).max(), 1)
    return points.astype("float32")


def quadruples(points, limit=150000):
    """Affine-invariant quad descriptors; sample large point sets reproducibly."""
    from math import comb

    count = len(points)
    if count < 4:
        return (np.empty((0, 3), "float32"), np.empty((0, 4), "int32"))
    if comb(count, 4) <= limit:
        ids = np.array(list(combinations(range(count), 4)), dtype=np.int32)
    else:
        random = np.random.default_rng(42)
        ids = np.sort(random.integers(0, count, (limit * 2, 4)), axis=1)
        ids = ids[(np.diff(ids, axis=1) > 0).all(1)][:limit]

        # Include all combinations of the strongest anchors in addition to the random
        # sample.
        ids = np.concatenate(
            [np.array(list(combinations(range(min(count, 28)), 4))), ids]
        )
    vertices, areas = (points[ids], [])
    for excluded in range(4):
        triangle = vertices[:, [j for j in range(4) if j != excluded]]
        a, b = (triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
        areas.append(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    areas = np.stack(areas, axis=1)
    order = np.argsort(areas, axis=1)
    areas = np.take_along_axis(areas, order, axis=1)
    ids = np.take_along_axis(ids, order, axis=1)

    # Discard nearly collinear quadruples: their area ratios are unstable.
    valid = areas[:, 0] > areas[:, -1] * 0.025
    features = areas / np.maximum(areas.sum(1, keepdims=True), 1e-10)
    return (features[valid, :3].astype("float32"), ids[valid])


def bright_stars(image, adaptive=False):
    """Detect saturated star centres and optionally add faint local-contrast peaks."""
    threshold = min(250, float(image.max()) * 0.98)
    _, _, stats, centers = cv2.connectedComponentsWithStats(
        (image >= threshold).astype("uint8")
    )
    saturated = centers[1:][(stats[1:, 4] >= 2) & (stats[1:, 4] <= 300)]
    if not adaptive:
        return saturated.astype("float32")
    smooth = cv2.GaussianBlur(image.astype("float32"), (0, 0), 1.0)
    contrast = smooth - cv2.GaussianBlur(smooth, (0, 0), 4.0)
    mask = (contrast == cv2.dilate(contrast, np.ones((7, 7), dtype="uint8"))) & (
        contrast > 12
    )
    mask[:3] = False
    mask[-3:] = False
    mask[:, :3] = False
    mask[:, -3:] = False
    y, x = np.where(mask)
    order = np.argsort(contrast[y, x])[::-1][:1800]
    x, y = (x[order], y[order])

    def offset(before, center, after):
        """Estimate a subpixel peak offset from three neighbouring intensity samples."""
        denominator = before - 2 * center + after
        return np.clip(
            0.5 * (before - after) / np.minimum(denominator, -1e-05), -0.5, 0.5
        )

    dx = offset(contrast[y, x - 1], contrast[y, x], contrast[y, x + 1])
    dy = offset(contrast[y - 1, x], contrast[y, x], contrast[y + 1, x])
    points = np.column_stack([x + dx, y + dy])
    if len(saturated):
        keep = []
        for start in range(0, len(points), 256):
            distances = np.linalg.norm(
                points[start : start + 256, None] - saturated[None], axis=2
            )
            keep.extend(distances.min(1) > 4)
        points = points[np.array(keep)]
    return np.concatenate([saturated, points]).astype("float32")


class PatternMatcher:
    """Propose reference constellations from scene stars and query evidence."""

    def __init__(self, directory, fast=False):
        """Load reference nodes and index their four-star descriptors with FLANN."""
        self.fast = fast
        self.names, self.points, features, self.quads, labels = ([], [], [], [], [])
        for file in sorted(Path(directory).glob("*_pattern.png")):
            points = read_stars(file)
            descriptor, ids = quadruples(points)
            label = len(self.names)
            self.names.append(file.stem.removesuffix("_pattern"))
            self.points.append(points)
            features.append(descriptor)
            self.quads.append(ids)
            labels.extend([label] * len(ids))
        self.labels = np.array(labels)
        self.all_quads = np.concatenate(self.quads)
        self.index = cv2.flann_Index(
            np.concatenate(features), dict(algorithm=1, trees=4)
        )

    def propose(self, candidates, image):
        """Find a small set of plausible figures for the initial matching pass."""
        ranked = sorted(
            [
                candidate
                for values in candidates.values()
                for candidate in values
                if candidate["appearance"] > 0.45
            ],
            key=lambda c: c["appearance"],
            reverse=True,
        )
        strong = [
            values[0]
            for values in candidates.values()
            if values
            and values[0]["score"] > 0.7
            and (
                values[0]["score"] - (values[1]["score"] if len(values) > 1 else 0)
                > 0.1
            )
        ]

        def anchors(stars, items):
            """Snap photographic candidates to distinct detected stars before geometric search."""
            indices = []
            for candidate in items:
                distance = np.linalg.norm(stars - np.array(candidate["xy"]), axis=1)
                nearest = int(distance.argmin())
                if distance[nearest] < 6 and nearest not in indices:
                    indices.append(nearest)
                if len(indices) >= 100:
                    break
            return stars[indices]

        stars = bright_stars(image)
        models = []
        if len(stars) >= 4:
            points = anchors(stars, ranked)
            if len(points) >= 4:
                models = self._search(points, stars, image.shape)
        adaptive = not models or max((model["image_score"] for model in models)) < 4

        # Faint stars may be missed when saturation is the only detection cue.
        if adaptive:
            stars = bright_stars(image, adaptive=True)
            if len(stars) >= 4:
                for items in [ranked, strong]:
                    points = anchors(stars, items)
                    if len(points) >= 4:
                        models.extend(self._search(points, stars, image.shape))
        for model in models:
            model["adaptive_stars"] = adaptive
            model["proposal_score"] = model["image_score"]
            if adaptive and strong:
                distance = np.linalg.norm(
                    np.array(model["projected"])[:, None]
                    - np.array([c["xy"] for c in strong])[None],
                    axis=2,
                ).min(0)
                model["proposal_score"] += 2 * float(np.exp(-(distance**2) / 8).sum())
        models.sort(key=lambda model: model["proposal_score"], reverse=True)
        if not models:
            return []
        selected = []
        for model in models:
            if model["proposal_score"] < max(2.0, 0.45 * models[0]["proposal_score"]):
                break
            if any(
                (
                    model["name"] == other["name"]
                    and np.mean(
                        np.linalg.norm(
                            np.array(model["projected"]) - other["projected"], axis=1
                        )
                    )
                    < 10
                    for other in selected
                )
            ):
                continue
            selected.append(model)
            if len(selected) == 4:
                break
        return selected

    def recover(self, candidates, image, fast=False, balanced=False):
        """Broaden weak matches using faint stars and several projective seeds."""
        stars = bright_stars(image, adaptive=True)
        if len(stars) < 4:
            return []
        ranked = sorted(
            [
                c
                for items in candidates.values()
                for c in items
                if c["appearance"] > 0.5
            ],
            key=lambda c: c["score"],
            reverse=True,
        )
        indices = []
        for candidate in ranked:
            distance = np.linalg.norm(stars - candidate["xy"], axis=1)
            nearest = int(distance.argmin())
            if distance[nearest] < 6 and nearest not in indices:
                indices.append(nearest)
            if len(indices) >= 65:
                break
        models = (
            self._search(
                stars[indices],
                stars,
                image.shape,
                exhaustive=True,
                seed_limit=20 if fast else 80,
            )
            if len(indices) >= 4
            else []
        )

        # Stop before projective recovery if the broader affine search already has
        # strong support.
        if balanced:
            from assignment import assign_figure

            for model in models:
                score, members = assign_figure(candidates, model)
                if (
                    score >= 1.25
                    and len(members) >= 8
                    and (model["matched_stars"] >= 8)
                ):
                    for candidate in models:
                        candidate["adaptive_stars"] = True
                        candidate["recovered"] = True
                    return models
        strong = [
            items[0]
            for items in candidates.values()
            if items
            and items[0]["score"] > 0.55
            and (
                items[0]["score"] - (items[1]["score"] if len(items) > 1 else 0) > 0.04
            )
        ]
        points = []
        for candidate in sorted(strong, key=lambda c: c["score"], reverse=True):
            if all(
                (
                    np.linalg.norm(np.array(candidate["xy"]) - point) > 8
                    for point in points
                )
            ):
                points.append(candidate["xy"])
        if len(points) >= 4 and (not fast):
            points = np.array(points, dtype="float32")

            # Reliable query locations can supply stars that the peak detector missed.
            keep = np.linalg.norm(stars[:, None] - points[None], axis=2).min(1) > 6
            augmented = np.concatenate([stars[keep], points])
            models.extend(
                self._search(
                    points, augmented, image.shape, exhaustive=True, projective=True
                )
            )
        for model in models:
            model["adaptive_stars"] = True
            model["recovered"] = True
        return models

    def _search(
        self, points, stars, shape, exhaustive=False, projective=False, seed_limit=80
    ):
        """Generate geometric seeds, rank image support and refine the best fits.

        The normal path retains the best affine seed for each pattern.
        Exhaustive recovery keeps several seeds, optionally initialized
        directly as projective transforms, before RANSAC refinement.
        """

        def distance_map(locations):
            """Store distance to the nearest supplied point at each scene pixel."""
            mask = np.ones(shape, dtype="uint8")
            xy = np.round(locations).astype(int)
            xy = np.clip(xy, [0, 0], [shape[1] - 1, shape[0] - 1])
            mask[xy[:, 1], xy[:, 0]] = 0
            return cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        query_map, star_map = (distance_map(points), distance_map(stars))
        features, quads = quadruples(points)
        if not len(features):
            return []
        neighbors_count = (
            128 if projective and len(points) <= 25 else 24 if exhaustive else 8
        )
        tolerance = (
            0.002 if neighbors_count == 128 else 0.00015 if exhaustive else 2.5e-05
        )
        neighbors, distances = self.index.knnSearch(
            features, neighbors_count, params={"checks": 128 if exhaustive else 64}
        )
        query, rank = np.where(distances < tolerance)
        reference = neighbors[query, rank]
        results = []
        for label, pattern in enumerate(self.points):
            selected = np.flatnonzero(self.labels[reference] == label)
            models = []
            homogeneous_pattern = np.column_stack([pattern, np.ones(len(pattern))])
            for start in range(0, len(selected), 256):
                batch = selected[start : start + 256]
                source = pattern[self.all_quads[reference[batch]]]
                target = points[quads[query[batch]]]
                homogeneous = np.concatenate(
                    [source, np.ones((*source.shape[:2], 1))], axis=2
                )

                # Four correspondences define each initial projective transform.
                if projective:
                    x0, y0 = (source[:, :, 0], source[:, :, 1])
                    u, v = (target[:, :, 0], target[:, :, 1])
                    z = np.zeros_like(x0)
                    o = np.ones_like(x0)
                    equations = np.stack(
                        [
                            np.stack([x0, y0, o, z, z, z, -u * x0, -u * y0], axis=2),
                            np.stack([z, z, z, x0, y0, o, -v * x0, -v * y0], axis=2),
                        ],
                        axis=2,
                    ).reshape(-1, 8, 8)
                    rhs = target.reshape(-1, 8, 1)
                    coefficients = (np.linalg.pinv(equations) @ rhs)[:, :, 0]
                    matrices = np.concatenate(
                        [coefficients, np.ones((len(coefficients), 1))], axis=1
                    ).reshape(-1, 3, 3)
                    mapped = homogeneous_pattern[None] @ matrices.transpose(0, 2, 1)
                    denominator = mapped[:, :, 2]
                    jacobian = (
                        matrices[:, :2, :2]
                        - matrices[:, :2, 2, None] * matrices[:, None, 2, :2]
                    )
                    singular = np.linalg.svd(jacobian, compute_uv=False)
                    valid = (denominator.min(1) > 0.2) & (denominator.max(1) < 5)
                    valid &= (
                        (singular[:, 1] > 150)
                        & (singular[:, 0] < 6500)
                        & (singular[:, 0] < 8 * singular[:, 1])
                    )
                    projected = mapped[:, :, :2] / np.maximum(
                        denominator[:, :, None], 1e-08
                    )
                    extent = np.ptp(projected, axis=1)
                    valid &= (extent.min(1) > 200) & (extent.max(1) < 6500)
                    projected, matrices = (projected[valid], matrices[valid])
                    if not len(matrices):
                        continue
                else:
                    matrices = np.linalg.pinv(homogeneous) @ target
                    errors = np.linalg.norm(
                        homogeneous @ matrices - target, axis=2
                    ).max(1)
                    singular = np.linalg.svd(matrices[:, :2], compute_uv=False)
                    valid = (
                        (errors < 20)
                        & (singular[:, 1] > 200)
                        & (singular[:, 0] < 6500)
                        & (singular[:, 0] < 8 * singular[:, 1])
                    )
                    matrices = matrices[valid]
                    if not len(matrices):
                        continue
                    projected = homogeneous_pattern[None] @ matrices
                x, y = (
                    projected[:, :, 0].astype("float32"),
                    projected[:, :, 1].astype("float32"),
                )
                near = cv2.remap(
                    query_map,
                    x,
                    y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=10000,
                )
                star_distance = cv2.remap(
                    star_map,
                    x,
                    y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=10000,
                )
                values = np.exp(
                    -(star_distance**2) / (2 * (5 if projective else 12) ** 2)
                ).sum(1)
                values += 0.15 * np.exp(-(near**2) / (2 * 15**2)).sum(1)
                for index in np.flatnonzero((near < 25).sum(1) >= 4):
                    models.append((values[index], matrices[index]))
                if len(models) > 1000:
                    models.sort(key=lambda model: model[0], reverse=True)
                    models = models[:200]
            models.sort(key=lambda model: model[0], reverse=True)
            seeds = []
            if not exhaustive:
                best = None
                for _, matrix in models[: 30 if self.fast else 100]:
                    for _ in range(3):
                        projected = homogeneous_pattern @ matrix
                        distances = np.linalg.norm(
                            projected[:, None] - stars[None], axis=2
                        )
                        nearest, distance = (distances.argmin(1), distances.min(1))
                        valid = distance < 15
                        if valid.sum() < 4:
                            break
                        matrix = np.linalg.lstsq(
                            homogeneous_pattern[valid],
                            stars[nearest[valid]],
                            rcond=None,
                        )[0]
                    homography = np.vstack([matrix.T, [0, 0, 1]])
                    score = self._evidence(pattern, homography, stars)[0]
                    if best is None or score > best[0]:
                        best = (score, homography)
                if best is not None:
                    seeds = [best[1]]
            else:
                for _, matrix in models[:seed_limit]:
                    homography = (
                        matrix if projective else np.vstack([matrix.T, [0, 0, 1]])
                    )
                    projected = cv2.perspectiveTransform(pattern[None], homography)[0]
                    if any(
                        (
                            np.mean(
                                np.linalg.norm(
                                    projected
                                    - cv2.perspectiveTransform(pattern[None], old)[0],
                                    axis=1,
                                )
                            )
                            < 10
                            for old in seeds
                        )
                    ):
                        continue
                    seeds.append(homography)

            # Refine using progressively tighter inlier distances while retaining the
            # best fit.
            for homography in seeds:
                best = None
                for tolerance in (
                    [45, 30, 20, 12, 8, 5] if exhaustive else [35, 25, 15, 10, 7]
                ):
                    score, projected, distance, nearest = self._evidence(
                        pattern, homography, stars, unique=exhaustive
                    )
                    if best is None or score > best[0]:
                        best = (score, homography.copy())
                    valid = distance < tolerance
                    if valid.sum() < 5 or len(set(nearest[valid])) < 5:
                        break
                    updated, inliers = cv2.findHomography(
                        pattern[valid],
                        stars[nearest[valid]],
                        cv2.RANSAC,
                        max(3, tolerance * 0.3) if exhaustive else 3,
                        maxIters=2000,
                    )
                    if updated is None or inliers.sum() < 5:
                        break
                    denominator = homogeneous_pattern @ updated[2]
                    if (
                        denominator.min() <= 0
                        or denominator.max() / denominator.min() > 4
                    ):
                        break
                    homography = updated
                score, projected, distance, _ = self._evidence(
                    pattern, homography, stars, unique=exhaustive
                )
                if exhaustive and score > best[0]:
                    best = (score, homography.copy())
                score, projected, distance, _ = self._evidence(
                    pattern, best[1], stars, unique=exhaustive
                )
                results.append(
                    dict(
                        name=self.names[label],
                        image_score=float(score),
                        homography=best[1].tolist(),
                        projected=projected.tolist(),
                        matched_stars=int((distance < 8).sum()),
                        total_stars=len(pattern),
                    )
                )
        return results

    @staticmethod
    def _evidence(pattern, homography, stars, unique=False):
        """Score unique image-star support for a projected reference constellation."""
        projected = cv2.perspectiveTransform(pattern[None], homography)[0]
        distances = np.linalg.norm(projected[:, None] - stars[None], axis=2)
        nearest, distance = (distances.argmin(1), distances.min(1))

        # Each image star can support only one reference node.
        used, evidence = (set(), 0.0)
        for node in np.argsort(distance):
            if int(nearest[node]) not in used:
                evidence += np.exp(-float(distance[node]) ** 2 / 32)
                used.add(int(nearest[node]))
            elif unique:
                distance[node] = 10000
        return (evidence - 0.1 * len(pattern), projected, distance, nearest)
