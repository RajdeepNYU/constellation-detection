

import numpy as np


def maximum_assignment(weights):
    """Solve maximum-weight one-to-one matching with optional unmatched rows.

    Zero-weight dummy columns allow a query to remain unmatched. The
    Hungarian solver minimizes negative weights and returns only positive
    real edges. Its working arrays use one-based indexing; column zero is
    the temporary root used while extending an augmenting path.
    """
    rows, columns = weights.shape

    # Negate weights for minimization and append one dummy option per query.
    cost = -np.column_stack([weights, np.zeros((rows, rows))])
    n, m = cost.shape

    # u and v are row/column potentials; p stores matched rows and way stores
    # predecessors.
    u, v = (np.zeros(n + 1), np.zeros(m + 1))
    p, way = (np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int))
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)

        # Grow the augmenting path until it reaches an unmatched column.
        while True:
            used[j0] = True
            i0 = p[j0]
            current = cost[i0 - 1] - u[i0] - v[1:]
            update = ~used[1:] & (current < minimum[1:])
            minimum[1:][update] = current[update]
            way[1:][update] = j0
            candidates = np.where(~used[1:], minimum[1:], np.inf)
            j1 = int(np.argmin(candidates)) + 1
            delta = minimum[j1]
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        # Follow predecessor columns backwards to update the matching.
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [
        (int(p[j] - 1), j - 1)
        for j in range(1, columns + 1)
        if p[j] and weights[p[j] - 1, j - 1] > 0
    ]


def assign_figure(candidates, model):
    """Match query candidates to projected figure nodes and score the joint fit.

    Return (joint_score, members), where members maps each selected query
    to its candidate dictionary and assignment weight.
    """
    projected, edges = (np.array(model["projected"]), [])
    for column, items in candidates.items():
        for candidate in items:
            if (
                candidate["appearance"] < 0.5
                or candidate["score"] < 0.25
                or candidate["score"] < items[0]["score"] - 0.12
            ):
                continue
            distances = np.linalg.norm(projected - np.array(candidate["xy"]), axis=1)
            node = int(distances.argmin())
            if distances[node] > 3.5:
                continue

            # Reward a close geometric match without promoting a much worse photographic
            # candidate.
            weight = (candidate["score"] - 0.25) * np.exp(-distances[node] ** 2 / 8)
            weight *= np.exp((candidate["score"] - items[0]["score"]) / 0.15)
            edges.append((float(weight), column, node, candidate))
    columns = list(candidates)
    weights = np.zeros((len(columns), len(projected)))
    matches = {}
    for weight, column, node, candidate in edges:
        row = columns.index(column)
        if weight > 0.02 and weight > weights[row, node]:
            weights[row, node] = weight
            matches[row, node] = candidate

    # Solve only rows and nodes that have an eligible edge; map indices back afterwards.
    active_rows = np.flatnonzero((weights > 0).any(axis=1))
    active_nodes = np.flatnonzero((weights > 0).any(axis=0))
    pairs = maximum_assignment(weights[np.ix_(active_rows, active_nodes)])
    selected = {
        columns[active_rows[row]]: (
            matches[active_rows[row], active_nodes[node]],
            float(weights[active_rows[row], active_nodes[node]]),
        )
        for row, node in pairs
    }
    score = sum((weight for _, weight in selected.values())) / np.sqrt(
        max(model["matched_stars"], 1)
    )
    score += 0.015 * model["image_score"]
    return (float(score), selected)
