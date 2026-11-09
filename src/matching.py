

import cv2
import numpy as np


class PatchMatcher:
    """Find image locations for 32 x 32 query patches.

    A half-resolution correlation search proposes locations; affine
    alignment and blur-aware texture checks verify them at full resolution.
    """

    def __init__(self, image, fast=False):
        """Prepare scene filters, circular masks and reusable projection bases."""
        self.fast = fast
        self.target_crops = {}
        self.image = image.astype("float32")
        filtered = cv2.GaussianBlur(self.image, (0, 0), 1)
        filtered -= cv2.GaussianBlur(filtered, (0, 0), 7)
        self.search_scale = 0.4 if fast else 0.5
        self.small = cv2.resize(
            filtered,
            None,
            fx=self.search_scale,
            fy=self.search_scale,
            interpolation=cv2.INTER_AREA,
        )
        y, x = np.mgrid[:21, :21]
        self.search_grids = []
        for scale in [0.85, 1.0, 1.15]:
            mask = (
                (x - 10) ** 2 + (y - 10) ** 2 < (12.8 * self.search_scale * scale) ** 2
            ).astype("float32")
            count = mask.sum()
            mean = cv2.matchTemplate(self.small, mask, cv2.TM_CCORR) / count
            variance = (
                cv2.matchTemplate(self.small**2, mask, cv2.TM_CCORR) - count * mean**2
            )
            self.search_grids.append(
                (scale, mask, np.sqrt(np.maximum(variance, 0.0001)))
            )
        y, x = np.mgrid[:32, :32]
        self.mask = (x - 15.5) ** 2 + (y - 15.5) ** 2 < 13**2
        self.basis = np.column_stack(
            [np.ones(self.mask.sum()), x[self.mask] - 15.5, y[self.mask] - 15.5]
        ).astype("float32")
        self.projection = np.linalg.pinv(self.basis)
        dx, dy = (x[self.mask] - 15.5, y[self.mask] - 15.5)
        radial = [np.ones(len(dx)), dx, dy, dx * dx + dy * dy]
        for sigma in [0.8, 1.3, 2.0, 3.0, 4.5, 7.0]:
            g = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
            radial.extend([g, dx * g, dy * g, (dx * dx - dy * dy) * g, dx * dy * g])
        self.radial_basis = np.linalg.qr(np.column_stack(radial))[0].astype("float32")
        self.padded = cv2.copyMakeBorder(
            self.image, 64, 64, 64, 64, cv2.BORDER_REFLECT_101
        )

    def correlation(self, a, b):
        """Compare masked patches after removing a fitted brightness plane."""
        a, b = (a[self.mask].copy(), b[self.mask].copy())

        # Remove a brightness plane so illumination gradients do not dominate
        # similarity.
        a -= self.basis @ (self.projection @ a)
        b -= self.basis @ (self.projection @ b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-08))

    def locate(self, patch):
        """Search rotations and scales, then verify the strongest distinct locations."""
        if patch.shape != (32, 32):
            raise ValueError("Expected the competition's 32x32 query patches")
        patch = patch.astype("float32")
        if patch.std() < 0.001:
            return []
        filtered = cv2.GaussianBlur(patch, (0, 0), 0.6)
        filtered -= cv2.GaussianBlur(filtered, (0, 0), 7)
        candidates = []
        for scale, mask, denominator in self.search_grids:
            for angle in range(0, 360, 30 if self.fast else 15):
                matrix = cv2.getRotationMatrix2D(
                    (15.5, 15.5), angle, scale * self.search_scale
                )
                matrix[:, 2] -= 5.5
                template = cv2.warpAffine(filtered, matrix, (21, 21))
                template = (template - (template * mask).sum() / mask.sum()) * mask
                response = cv2.matchTemplate(self.small, template, cv2.TM_CCORR)
                response /= denominator * np.linalg.norm(template) + 1e-08

                # Suppress flat-region peaks caused by round-off in correlation.
                response[denominator < np.sqrt(mask.sum())] = -1
                for _ in range(2 if self.fast else 4):
                    _, score, _, (x, y) = cv2.minMaxLoc(response)
                    candidates.append(
                        (
                            score,
                            (np.array([x + 10.0, y + 10.0]) + 0.5) / self.search_scale
                            - 0.5,
                            angle,
                            scale,
                        )
                    )
                    response[max(y - 8, 0) : y + 9, max(x - 8, 0) : x + 9] = -1
        candidates.sort(key=lambda item: item[0], reverse=True)
        distinct = []
        for candidate in candidates:
            if all(
                (
                    np.linalg.norm(candidate[1] - previous[1]) > 10
                    for previous in distinct
                )
            ):
                distinct.append(candidate)
            if len(distinct) >= (24 if self.fast else 48):
                break
        return self.verify(patch, distinct)

    def verify(self, patch, candidates, limit=8):
        """Refine candidate poses with ECC and score their appearance and texture.

        Each input pose is (coarse_score, centre_xy, angle, scale). Returned
        candidates contain score, appearance and the refined scene centre.
        """
        refined = []
        query = cv2.GaussianBlur(patch, (0, 0), 0.5)
        query_high = query - cv2.GaussianBlur(query, (0, 0), 2)
        for _, center, angle, scale in candidates:
            left, top = np.round(center).astype(int) - 48
            crop = self.padded[top + 64 : top + 160, left + 64 : left + 160]
            cached_blurs = self.target_crops.get((left, top))
            matrix = cv2.getRotationMatrix2D((0, 0), angle, scale)[:, :2]
            warp = np.column_stack(
                [matrix, center - [left, top] - matrix @ [15.5, 15.5]]
            ).astype("float32")
            try:
                _, warp = cv2.findTransformECC(
                    patch,
                    (
                        cached_blurs[0.8]
                        if cached_blurs is not None
                        else cv2.GaussianBlur(crop, (0, 0), 0.8)
                    ),
                    warp,
                    cv2.MOTION_AFFINE,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 0.0001),
                    None,
                    5,
                )
            except cv2.error:
                continue
            scales = np.linalg.svd(warp[:, :2], compute_uv=False)
            if (
                scales.min() < 0.6
                or scales.max() > 1.5
                or scales.max() / scales.min() > 1.2
            ):
                continue
            position = warp @ [15.5, 15.5, 1] + [left, top]
            if not (
                0 <= position[0] < self.image.shape[1]
                and 0 <= position[1] < self.image.shape[0]
            ):
                continue
            best_score = -1.0
            best_appearance = -1.0
            for blur in [0.4, 0.8, 1.2, 1.6, 2.0]:
                source = (
                    cached_blurs[blur]
                    if cached_blurs is not None
                    else cv2.GaussianBlur(crop, (0, 0), blur)
                )
                restored = cv2.warpAffine(
                    source,
                    warp,
                    (32, 32),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                )
                restored = cv2.GaussianBlur(restored, (0, 0), 0.5)
                raw_score = self.correlation(restored, query)
                high_score = self.correlation(
                    restored - cv2.GaussianBlur(restored, (0, 0), 2), query_high
                )
                a, b = (restored[self.mask].copy(), query[self.mask].copy())

                # Suppress the generic round-star profile and compare the remaining
                # local texture.
                a -= self.radial_basis @ (self.radial_basis.T @ a)
                b -= self.radial_basis @ (self.radial_basis.T @ b)
                texture_score = float(
                    a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-08)
                )
                best_score = max(
                    best_score, 0.2 * raw_score + 0.2 * high_score + 0.6 * texture_score
                )
                best_appearance = max(
                    best_appearance, 0.5 * raw_score + 0.5 * high_score
                )
            refined.append(
                dict(
                    score=float(best_score),
                    appearance=float(best_appearance),
                    xy=position.tolist(),
                )
            )
        refined.sort(key=lambda item: item["score"], reverse=True)
        unique = []
        for candidate in refined:
            if all(
                (
                    np.linalg.norm(np.array(candidate["xy"]) - previous["xy"]) > 12
                    for previous in unique
                )
            ):
                unique.append(candidate)
        if len(unique) <= limit:
            return unique

        # Keep ambiguous bright-star matches too; later geometry may disambiguate them.
        selected = unique[:limit]
        for candidate in sorted(
            unique, key=lambda item: item["appearance"], reverse=True
        )[:limit]:
            if candidate not in selected:
                selected.append(candidate)
        return sorted(selected, key=lambda item: item["score"], reverse=True)

    def descriptor(self, patch):
        """Combine normalized intensity and high-pass channels for targeted lookup."""
        smoothed = cv2.GaussianBlur(patch.astype("float32"), (0, 0), 0.5)
        vectors = []
        for channel in [smoothed, smoothed - cv2.GaussianBlur(smoothed, (0, 0), 2)]:
            vector = channel[self.mask].copy()
            vector -= self.basis @ (self.projection @ vector)
            vectors.append(vector / max(np.linalg.norm(vector), 1e-08))
        return np.concatenate(vectors) / np.sqrt(2)

    def prepare_targets(self, positions):
        """Build pose descriptors near projected figure stars and cache blurred crops."""
        descriptors, poses = ([], [])
        for center in positions:
            left, top = np.round(center).astype(int) - 48
            crop = self.padded[top + 64 : top + 160, left + 64 : left + 160]
            if crop.shape != (96, 96):
                continue
            if (left, top) not in self.target_crops:
                self.target_crops[left, top] = {
                    blur: cv2.GaussianBlur(crop, (0, 0), blur)
                    for blur in [0.4, 0.8, 1.2, 1.6, 2.0]
                }
            crop = self.target_crops[left, top][0.8]
            for scale in [0.85, 1.0, 1.15]:
                for angle in range(0, 360, 15):
                    matrix = cv2.getRotationMatrix2D((0, 0), angle, scale)[:, :2]
                    warp = np.column_stack(
                        [matrix, center - [left, top] - matrix @ [15.5, 15.5]]
                    ).astype("float32")
                    template = cv2.warpAffine(
                        crop,
                        warp,
                        (32, 32),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    )
                    descriptors.append(self.descriptor(template))
                    poses.append((center, angle, scale))
        if not descriptors:
            return None
        return (np.array(descriptors, dtype="float32"), poses)

    def targeted(self, patch, library):
        """Recheck a query at figure-guided locations using several orientation sectors."""
        if library is None:
            return []
        descriptors, poses = library
        scores = descriptors @ self.descriptor(patch)
        candidates = []
        for start in range(0, len(poses), 72):

            # Nearly circular stars have uncertain orientation; keep an option in each
            # sector.
            sectors = 3 if self.fast else 6
            width = 24 // sectors
            for sector in range(sectors):
                indices = [
                    start + scale * 24 + angle
                    for scale in range(3)
                    for angle in range(sector * width, (sector + 1) * width)
                ]
                index = max(indices, key=lambda i: scores[i])
                center, angle, scale = poses[index]
                candidates.append((float(scores[index]), center, angle, scale))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return self.verify(patch.astype("float32"), candidates, limit=48)
