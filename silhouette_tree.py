"""Adaptive Silhouette Tree (AST): semantic IDs as paths in an adaptive divisive tree.

Paper: "Not All Branches Are Equal: Adaptive Semantic ID Construction for
Generative Recommendation" (RecSys '26), Section 4.

An item's semantic ID is ``(s^0, ..., s^{m-1})``. The first ``L = m - 1`` tokens are
the path of the item in a top-down KMeans tree whose branching factor ``K_v`` is chosen
per node by maximizing the silhouette coefficient (Sec. 4.2). The last token is a
deterministic, collision-free suffix inside each leaf bucket.

The constrained tokenizer (Sec. 4.4.3, mass-aware constrained refinement) requires every
child to carry at least a fraction ``rho`` (``min_share_prob``) of its parent's mass. It uses a
mass-weighted KMeans initialization, alternates {exact MILP assignment (Eq. 1-4), mass-weighted
center update} rounds, and scores splits with the mass-weighted silhouette:
``milp_refine_rounds=3, milp_use_weighted_centers=True, weighted_silhouette=True``.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def _regular_kmeans_labels(X: np.ndarray, K: int, *, seed: int, max_iter: int, n_init: int) -> np.ndarray:
    km = KMeans(
        n_clusters=int(K),
        random_state=int(seed),
        max_iter=int(max_iter),
        n_init=int(n_init),
        init="k-means++",
    )
    return km.fit_predict(X).astype(np.int32)


def _token_ids(tokenizer, flat_tokens: list[str]) -> list[int]:
    """Ids of the ``<cℓ_k>`` tokens, in the same order as ``flat_tokens`` (added if missing)."""
    tokenizer.add_tokens(flat_tokens, special_tokens=False)
    ids = tokenizer.convert_tokens_to_ids(flat_tokens)
    if any(i is None or int(i) < 0 for i in ids):
        raise ValueError("Some tokens could not be mapped to ids (UNK).")
    return [int(i) for i in ids]


def encode_items(
    id2text: list[str],
    *,
    encoder_name: str,
    batch_size: int = 8,
    max_seq_length: int = 512,
) -> np.ndarray:
    """L2-normalized sentence embeddings of the item texts, shape [N, d]."""
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(encoder_name)
    enc.max_seq_length = max_seq_length
    return enc.encode(
        id2text,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def _exact_assignment_milp_fixed_centers(
    X: np.ndarray,
    p_local: np.ndarray,
    centers: np.ndarray,
    *,
    x_min_mass: float,
    candidate_topk: int | None,
    warm_labels: np.ndarray | None,
    time_limit: float | None,
    mip_gap: float,
    solver_name: str,
    verbose: bool,
) -> np.ndarray:
    """
    Exact hard assignment w.r.t. fixed centers (paper Eq. 1-4):

        min  sum_{i,j} ||e_i - mu_j||^2 z_ij
        s.t. sum_j z_ij = 1                 for all i
             sum_i pi_i z_ij >= rho         for all j
             z_ij in {0, 1}

    Solved with python-mip (CBC by default, Gurobi optional). With ``candidate_topk``,
    item i may only be assigned to its ``candidate_topk`` nearest centers.
    Returns labels in [0, K-1].
    """
    p = np.asarray(p_local, dtype=np.float64).reshape(-1)
    if np.any(p < 0):
        raise ValueError("p_local must be nonnegative")
    s = float(p.sum())
    if s <= 0:
        raise ValueError("p_local must sum to >0")
    p = p / s

    X = np.asarray(X, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    Nn = int(X.shape[0])
    Kk = int(centers.shape[0])

    if Kk * float(x_min_mass) > 1.0 + 1e-12:
        raise RuntimeError(f"Infeasible: K*x_min_mass={Kk * float(x_min_mass):.6f} > 1")

    D = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)

    if candidate_topk is None or int(candidate_topk) >= Kk:
        cand_js = [np.arange(Kk, dtype=int) for _ in range(Nn)]
    else:
        topk = int(candidate_topk)
        if topk <= 0:
            raise ValueError("candidate_topk must be >= 1")
        idx = np.argpartition(D, kth=topk - 1, axis=1)[:, :topk]
        rows = np.arange(Nn)[:, None]
        order = np.argsort(D[rows, idx], axis=1)
        idx = idx[rows, order]
        cand_js = [idx[i].astype(int) for i in range(Nn)]

    reachable = np.zeros(Kk, dtype=np.float64)
    for i in range(Nn):
        reachable[cand_js[i]] += p[i]
    if float(reachable.min()) < float(x_min_mass) - 1e-12:
        raise RuntimeError(
            f"Infeasible under candidate_topk={candidate_topk}: "
            f"min_reachable={float(reachable.min()):.6f} < x_min_mass={float(x_min_mass):.6f}"
        )

    try:
        import mip
        from mip import BINARY, Model, OptimizationStatus, xsum
    except Exception as e:
        raise ImportError("MILP requires python-mip (pip install mip)") from e

    solver = solver_name.upper()
    if solver == "CBC":
        solver_enum = mip.CBC
    elif solver in ("GRB", "GUROBI"):
        solver_enum = mip.GRB
    else:
        raise ValueError("milp_solver_name must be 'CBC' or 'GRB'")

    model = Model(sense=mip.MINIMIZE, solver_name=solver_enum)
    model.verbose = int(bool(verbose))
    if time_limit is not None:
        model.max_seconds = float(time_limit)
    model.max_mip_gap = float(mip_gap)

    edges_by_i = [[] for _ in range(Nn)]
    edges_by_j = [[] for _ in range(Kk)]
    z = {}
    for i in range(Nn):
        for j in cand_js[i]:
            j = int(j)
            z[(i, j)] = model.add_var(var_type=BINARY, name=f"z_{i}_{j}")
            edges_by_i[i].append((i, j))
            edges_by_j[j].append((i, j))

    model.objective = xsum(float(D[i, j]) * z[(i, j)] for (i, j) in z.keys())
    for i in range(Nn):
        model += xsum(z[e] for e in edges_by_i[i]) == 1.0
    for j in range(Kk):
        model += xsum(float(p[i]) * z[(i, j)] for (i, j) in edges_by_j[j]) >= float(x_min_mass)

    if warm_labels is not None:
        wl = np.asarray(warm_labels, dtype=int)
        if wl.shape[0] == Nn:
            start = [(z[(i, int(wl[i]))], 1.0) for i in range(Nn) if (i, int(wl[i])) in z]
            if start:
                model.start = start

    status = model.optimize()
    if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
        raise RuntimeError(f"python-mip status={status}")

    labels = np.full(Nn, -1, dtype=np.int32)
    for (i, j), var in z.items():
        if var.x is not None and var.x > 0.5:
            labels[i] = int(j)
    if np.any(labels < 0):
        raise RuntimeError("MILP produced incomplete assignment")
    return labels


def build_item2tokseq_ids_silhouette_tree(
    id2text: list[str],
    tokenizer,
    *,
    base: int = 4096,                    # B: max branching factor / per-level vocabulary
    m: int = 4,                          # semantic ID length (L = m-1 tree levels + 1 suffix token)
    seed: int = 42,
    encoder_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 8,
    max_seq_length: int = 512,

    # optional precomputed inputs
    item_embs: np.ndarray | None = None,       # [N, d]; re-normalized to unit length
    item_embs_normalized: bool = False,        # True: item_embs come from encode_items (skip re-normalization)
    item_probs: np.ndarray | None = None,      # [N] item mass p_i (None -> uniform)

    # KMeans / silhouette model selection
    max_iter_kmeans: int = 50,
    n_init_kmeans: int = 10,
    silhouette_metric: str = "euclidean",
    patience: int = 5,
    silhouette_sample_size: int | None = 2000,
    k_search_step: int = 1,
    weighted_silhouette: bool = True,          # mass-weighted silhouette; needs item_probs

    # tree behaviour
    fixed_k: int | list[int] | str | None = None,
    density_driven_k: bool = False,
    target_items_per_leaf: float = 3.0,
    k_window_delta: int = 2,
    k_cap: int = 64,
    min_items_to_split: int = 3,
    adaptive_depth: bool = False,
    max_depth: int = 6,
    max_items_per_leaf: int = 4,
    capacity_aware_min_k: bool = True,
    stop_recursion_if_no_split: bool = True,

    # minimum-mass constraint (rho)
    min_share_prob: float | None = None,
    enforce_min_share_hard: bool = True,

    # exact constrained assignment (MILP)
    MILP: bool = False,
    milp_only_if_needed: bool = True,
    milp_candidate_topk: int | None = 16,
    milp_time_limit: float | None = 60.0,
    milp_gap: float = 1e-3,
    milp_solver_name: str = "CBC",
    milp_verbose: bool = False,
    milp_use_weighted_centers: bool = True,    # mass-weighted KMeans init + center updates
    milp_refine_rounds: int = 3,               # 1 = one assignment w.r.t. fixed centers; >1 = alternating refinement

    # logging
    print_each_clustering: bool = True,
    print_stats: bool = True,
    show_examples: int = 5,
    return_tree_stats: bool = False,
) -> dict[int, list[int]] | tuple[dict[int, list[int]], dict]:
    """
    Build AST semantic IDs. Returns ``item_id -> [token_id] * m`` (unique per item).

    1) Encode items to unit-norm embeddings (unless ``item_embs`` is given).
    2) For the ``L = m - 1`` prefix levels, recursively split each node with the K in
       ``[K_min, K_max]`` that maximizes the silhouette (early stop after ``patience``
       non-improving candidates). ``K_min`` is capacity-aware so every final bucket fits
       in ``base`` suffix tokens; with ``rho`` set, ``K_max <= floor(1/rho)`` and splits
       violating the mass constraint are repaired by MILP or rejected.
    3) The last token enumerates items inside each prefix bucket (collision-free).
    """
    N = len(id2text)
    if N == 0:
        return {}
    if base < 2:
        raise ValueError("base must be >= 2.")
    if m < 2:
        raise ValueError("m must be >= 2 (need at least one prefix layer + one suffix layer).")
    if int(k_search_step) < 1:
        raise ValueError("k_search_step must be >= 1.")
    if int(milp_refine_rounds) < 1:
        raise ValueError("milp_refine_rounds must be >= 1.")
    if min_share_prob is not None:
        min_share_prob = float(min_share_prob)
        if not (0.0 < min_share_prob < 1.0):
            raise ValueError("min_share_prob must be in (0,1) or None.")

    L = m - 1  # number of prefix levels

    # ------------------------------------------------------------------
    # 0) item mass p_i (global, normalized); None -> uniform within each node
    # ------------------------------------------------------------------
    if item_probs is not None:
        p_items = np.asarray(item_probs, dtype=np.float64).reshape(-1)
        if p_items.shape[0] != N:
            raise ValueError(f"item_probs length {p_items.shape[0]} must equal N={N}")
        if np.any(p_items < 0):
            raise ValueError("item_probs must be non-negative")
        s = float(p_items.sum())
        if s <= 0:
            raise ValueError("item_probs must sum to a positive value")
        p_items = p_items / s
    else:
        p_items = None

    # ------------------------------------------------------------------
    # 1) text -> unit-norm embeddings
    # ------------------------------------------------------------------
    if item_embs is None:
        embs = encode_items(id2text, encoder_name=encoder_name, batch_size=batch_size, max_seq_length=max_seq_length)
    else:
        embs = np.asarray(item_embs, dtype=np.float32)
        if embs.ndim != 2:
            raise ValueError(f"item_embs must be 2D [N,d], got shape={embs.shape}")
        if embs.shape[0] != N:
            raise ValueError(f"item_embs.shape[0]={embs.shape[0]} must equal len(id2text)={N}")
        if not item_embs_normalized:
            norms = np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
            embs = (embs / norms).astype(np.float32)

    if not np.isfinite(embs).all():
        bad = int(np.size(embs) - np.isfinite(embs).sum())
        raise ValueError(f"Embeddings contain non-finite values (NaN/Inf): bad={bad}")

    # ------------------------------------------------------------------
    # 2) token ids for all m levels: level l uses its own vocabulary <cl_0> .. <cl_{B-1}>
    # ------------------------------------------------------------------
    num_token_levels = max(m, int(max_depth) + 1) if adaptive_depth else m
    flat_tokens = [f"<c{lvl}_{i}>" for lvl in range(num_token_levels) for i in range(base)]
    flat_ids = _token_ids(tokenizer, flat_tokens)
    tok_ids = [flat_ids[lvl * base:(lvl + 1) * base] for lvl in range(num_token_levels)]

    # ------------------------------------------------------------------
    # 3) prefix codes, built top-down
    # ------------------------------------------------------------------
    k_schedule: list[int] | None = None
    if fixed_k is not None:
        if isinstance(fixed_k, (list, tuple)):
            k_schedule = [int(x) for x in fixed_k]
        elif isinstance(fixed_k, str) and ("," in fixed_k or "-" in fixed_k):
            sep = "," if "," in fixed_k else "-"
            k_schedule = [int(x.strip()) for x in fixed_k.split(sep)]
        else:
            k_schedule = [int(fixed_k)] * max(L, int(max_depth) if adaptive_depth else L)
        if print_stats:
            print(f"[AST] Using layer-wise branching schedule: {k_schedule}")

    codes = [np.zeros(N, dtype=np.int32) for _ in range(L)]
    split_counts = np.zeros(L, dtype=np.int64)
    chosen_k_hist = [defaultdict(int) for _ in range(L)]
    node_counts = np.zeros(L, dtype=np.int64)
    leaf_counts = np.zeros(L, dtype=np.int64)
    split_records: list[dict] = []
    split_sils_by_level: dict[int, list[float]] = defaultdict(list)

    def _build_tree_stats() -> dict:
        all_sils = [r["silhouette"] for r in split_records if np.isfinite(r["silhouette"])]
        return {
            "silhouette_overall_mean": round(float(np.mean(all_sils)), 4) if all_sils else None,
            "silhouette_overall_median": round(float(np.median(all_sils)), 4) if all_sils else None,
            "silhouette_overall_std": round(float(np.std(all_sils)), 4) if all_sils else None,
            "silhouette_overall_min": round(float(np.min(all_sils)), 4) if all_sils else None,
            "silhouette_overall_max": round(float(np.max(all_sils)), 4) if all_sils else None,
            "silhouette_by_level": {
                str(lvl): {
                    "count": len(sils),
                    "mean": round(float(np.mean(sils)), 4),
                    "median": round(float(np.median(sils)), 4),
                    "std": round(float(np.std(sils)), 4) if len(sils) > 1 else 0.0,
                    "min": round(float(np.min(sils)), 4),
                    "max": round(float(np.max(sils)), 4),
                }
                for lvl, sils in sorted(split_sils_by_level.items())
            },
            "split_records": split_records,
        }

    def _silhouette(X, labels, *, sample_size, seed_for_sample, weights=None) -> float:
        """(Weighted) mean silhouette of a split; NaN if undefined.

        For large nodes it is estimated on ``sample_size`` items, drawn proportionally
        to the weights when given (unbiased estimator of the weighted silhouette).
        """
        try:
            labels = np.asarray(labels, dtype=np.int32)
            n_local = int(X.shape[0])
            if np.unique(labels).size < 2 or n_local <= np.unique(labels).size:
                return float("nan")
            if not np.isfinite(X).all():
                return float("nan")

            w = None
            if weights is not None:
                w = np.asarray(weights, dtype=np.float64).reshape(-1)
                if w.shape[0] != n_local:
                    raise ValueError("weights length mismatch in silhouette computation")
                if np.any(w < 0):
                    raise ValueError("weights must be nonnegative")
                sw = float(w.sum())
                w = None if sw <= 0 else w / sw

            if sample_size is not None and n_local > int(sample_size):
                rng = np.random.default_rng(int(seed_for_sample))
                if w is None:
                    idx = rng.choice(n_local, size=int(sample_size), replace=False)
                else:
                    idx = rng.choice(n_local, size=int(sample_size), replace=False, p=w)
                Xs, ys = X[idx], labels[idx]
                if np.unique(ys).size < 2:
                    return float("nan")
                sil_vals = silhouette_samples(Xs, ys, metric=silhouette_metric)
                if w is None:
                    val = float(np.mean(sil_vals))
                else:
                    ws = w[idx]
                    ws_sum = float(ws.sum())
                    if ws_sum <= 0:
                        return float("nan")
                    val = float(np.sum((ws / ws_sum) * sil_vals))
            else:
                sil_vals = silhouette_samples(X, labels, metric=silhouette_metric)
                val = float(np.mean(sil_vals)) if w is None else float(np.sum(w * sil_vals))

            return val if np.isfinite(val) else float("nan")
        except Exception as e:
            print(f"[silhouette] {type(e).__name__}: {e}")
            return float("nan")

    def _capacity_lower_bound_k(n_items: int, level: int) -> int:
        # K >= n / B^(L - level) so that, in principle, leaf buckets fit in B suffix tokens.
        if not capacity_aware_min_k:
            return 2
        lb = int(math.ceil(float(n_items) / float(base ** (L - level))))
        return max(2, lb)

    def _split_stats(item_ids_local: np.ndarray, labels_raw: np.ndarray) -> dict:
        labels_raw = np.asarray(labels_raw, dtype=np.int32)
        if labels_raw.ndim != 1 or labels_raw.shape[0] != item_ids_local.shape[0]:
            raise ValueError("labels shape mismatch in _split_stats")
        uniq, inv = np.unique(labels_raw, return_inverse=True)
        labels = inv.astype(np.int32, copy=False)
        K_eff = int(uniq.size)
        child_counts = np.bincount(labels, minlength=K_eff).astype(np.int64)

        if p_items is not None:
            parent_weights = p_items[item_ids_local]
            child_masses_global = np.bincount(labels, weights=parent_weights, minlength=K_eff).astype(np.float64)
            parent_mass_global = float(parent_weights.sum())
        else:
            child_masses_global = child_counts.astype(np.float64) / float(N)
            parent_mass_global = float(item_ids_local.size) / float(N)

        if parent_mass_global > 0:
            child_shares_parent = child_masses_global / parent_mass_global
        else:
            child_shares_parent = np.zeros_like(child_masses_global, dtype=np.float64)

        return {
            "labels": labels,
            "K_eff": K_eff,
            "child_counts": child_counts,
            "parent_mass_global": parent_mass_global,
            "min_share_parent": float(child_shares_parent.min()) if K_eff > 0 else 0.0,
            "max_share_parent": float(child_shares_parent.max()) if K_eff > 0 else 0.0,
            "min_prob_global": float(child_masses_global.min()) if K_eff > 0 else 0.0,
        }

    def _node_local_probs(item_ids_local: np.ndarray) -> np.ndarray:
        """pi_i = p_i / sum_{r in S_v} p_r (uniform when masses are unavailable)."""
        n_local = int(item_ids_local.size)
        if n_local <= 0:
            raise ValueError("empty node in _node_local_probs")
        if p_items is None:
            return np.full(n_local, 1.0 / n_local, dtype=np.float64)
        p_local = np.asarray(p_items[item_ids_local], dtype=np.float64)
        s = float(p_local.sum())
        if s <= 0:
            return np.full(n_local, 1.0 / n_local, dtype=np.float64)
        return p_local / s

    def _centers_from_labels(X, p_local, labels, K) -> np.ndarray | None:
        """Center update: mass-weighted mean or plain mean. None if a cluster is empty."""
        X = np.asarray(X, dtype=np.float64)
        p_local = np.asarray(p_local, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        centers = np.zeros((K, int(X.shape[1])), dtype=np.float64)
        for j in range(K):
            mask = labels == j
            if not np.any(mask):
                return None
            Xj = X[mask]
            if milp_use_weighted_centers:
                wj = p_local[mask]
                sw = float(wj.sum())
                centers[j] = Xj.mean(axis=0) if sw <= 0 else np.sum(Xj * (wj[:, None] / sw), axis=0)
            else:
                centers[j] = Xj.mean(axis=0)
        return centers

    def _milp_labels(X, item_ids_local, *, K_try: int, seed_local: int, warm_labels=None) -> np.ndarray | None:
        """
        Constrained split for a fixed K:
          1) centers from (mass-weighted) KMeans
          2) exact MILP assignment w.r.t. the current centers
          3) recompute centers from the assignment and repeat, up to
             ``milp_refine_rounds`` assignments or until assignments/centers stop changing.
        Returns None if MILP is disabled, infeasible, or the solver fails.
        """
        if not MILP or min_share_prob is None:
            return None
        n_local = int(X.shape[0])
        if n_local <= 1 or K_try < 2 or K_try >= n_local:
            return None
        x_min_mass = float(min_share_prob)
        if K_try * x_min_mass > 1.0 + 1e-12:
            return None

        p_local = _node_local_probs(item_ids_local)

        km = KMeans(
            n_clusters=int(K_try),
            random_state=int(seed_local),
            n_init=int(n_init_kmeans),
            max_iter=int(max_iter_kmeans),
        )
        if milp_use_weighted_centers:
            km.fit(X, sample_weight=p_local)
        else:
            km.fit(X)
        centers = np.asarray(km.cluster_centers_, dtype=np.float64)

        warm = None
        if warm_labels is not None:
            warm_labels = np.asarray(warm_labels, dtype=np.int32)
            if warm_labels.shape[0] == n_local:
                warm = warm_labels

        prev_labels = None
        try:
            for _ in range(int(milp_refine_rounds)):
                labels = _exact_assignment_milp_fixed_centers(
                    X=X,
                    p_local=p_local,
                    centers=centers,
                    x_min_mass=x_min_mass,
                    candidate_topk=milp_candidate_topk,
                    warm_labels=warm,
                    time_limit=milp_time_limit,
                    mip_gap=float(milp_gap),
                    solver_name=str(milp_solver_name),
                    verbose=bool(milp_verbose),
                )
                if labels.shape[0] != n_local:
                    return None
                if prev_labels is not None and np.array_equal(labels, prev_labels):
                    return labels

                new_centers = _centers_from_labels(X, p_local, labels, int(K_try))
                if new_centers is None:
                    return labels
                if np.allclose(new_centers, centers, rtol=0.0, atol=1e-10):
                    return labels

                prev_labels = labels.copy()
                centers = new_centers
                warm = labels
            return prev_labels if prev_labels is not None else labels
        except Exception as e:
            if print_each_clustering or print_stats:
                print(f"[MILP] node solve failed (n={n_local}, K={K_try}): {type(e).__name__}: {e}")
            return None

    def _pick_best_k(X, item_ids_local, *, K_min: int, K_max: int, seed_local: int):
        """Silhouette-maximizing K in [K_min, K_max] (step ``k_search_step``) with patience."""
        if K_max < K_min:
            return 1, None, None
        if min_share_prob is not None:
            K_max = min(int(K_max), int(math.floor(1.0 / float(min_share_prob))))
            if K_max < K_min:
                return 1, None, None

        best = None
        best_sil = -np.inf
        no_improve = 0
        sil_weights = _node_local_probs(item_ids_local) if (weighted_silhouette and p_items is not None) else None

        for K_try in range(int(K_min), int(K_max) + 1, int(k_search_step)):
            labels_regular = _regular_kmeans_labels(
                X, int(K_try), seed=int(seed_local), max_iter=max_iter_kmeans, n_init=n_init_kmeans
            )
            diag_regular = _split_stats(item_ids_local, labels_regular)
            K_eff_regular = int(diag_regular["K_eff"])
            if K_eff_regular < 2 or int(X.shape[0]) <= K_eff_regular:
                continue

            chosen_labels = np.asarray(diag_regular["labels"], dtype=np.int32)
            chosen_diag = diag_regular
            chosen_diag["assign_method"] = "kmeans"

            regular_satisfies = True
            if enforce_min_share_hard and (min_share_prob is not None):
                regular_satisfies = bool(diag_regular["min_share_parent"] + 1e-15 >= float(min_share_prob))

            if MILP and (min_share_prob is not None) and ((not milp_only_if_needed) or (not regular_satisfies)):
                labels_milp = _milp_labels(
                    X,
                    item_ids_local,
                    K_try=int(K_try),
                    seed_local=int(seed_local) + 1009 * int(K_try),
                    warm_labels=chosen_labels,
                )
                if labels_milp is not None:
                    diag_milp = _split_stats(item_ids_local, labels_milp)
                    K_eff_milp = int(diag_milp["K_eff"])
                    if K_eff_milp >= 2 and int(X.shape[0]) > K_eff_milp:
                        chosen_labels = np.asarray(diag_milp["labels"], dtype=np.int32)
                        chosen_diag = diag_milp
                        chosen_diag["assign_method"] = "milp"

            if enforce_min_share_hard and (min_share_prob is not None):
                if chosen_diag["min_share_parent"] + 1e-15 < float(min_share_prob):
                    continue

            sil_val = _silhouette(
                X,
                chosen_labels,
                sample_size=silhouette_sample_size,
                seed_for_sample=int(seed_local) + 99991 + int(X.shape[0]) + 13 * int(K_try),
                weights=sil_weights,
            )
            if not np.isfinite(sil_val):
                continue
            chosen_diag["silhouette"] = float(sil_val)

            # strictly better silhouette, or a tie broken toward the smaller K
            if (sil_val > best_sil + 1e-12) or (
                abs(sil_val - best_sil) <= 1e-12 and (best is None or int(chosen_diag["K_eff"]) < int(best["K_eff"]))
            ):
                best_sil = float(sil_val)
                best = chosen_diag
                no_improve = 0
            else:
                no_improve += 1
                if patience is not None and int(patience) >= 0 and no_improve >= int(patience):
                    break

        if best is None:
            return 1, None, None
        return int(best["K_eff"]), np.asarray(best["labels"], dtype=np.int32), best

    def _recurse(level: int, item_ids: np.ndarray, *, node_seed: int) -> None:
        if level >= L:
            return
        node_counts[level] += 1
        n = int(item_ids.size)
        if n <= 1:
            leaf_counts[level] += 1
            return
        if density_driven_k and n <= int(min_items_to_split):
            leaf_counts[level] += 1
            return

        X = embs[item_ids]
        if k_schedule is not None:
            K_level = k_schedule[level] if level < len(k_schedule) else k_schedule[-1]
            K_target = min(int(K_level), n - 1)
            min_k_eff = _capacity_lower_bound_k(n, level)
            if K_target < max(2, min_k_eff):
                leaf_counts[level] += 1
                return
            labels_raw = _regular_kmeans_labels(
                X, int(K_target), seed=int(node_seed), max_iter=max_iter_kmeans, n_init=n_init_kmeans
            )
            diag = _split_stats(item_ids, labels_raw)
            K = int(diag["K_eff"])
            labels = np.asarray(diag["labels"], dtype=np.int32)
            diag["assign_method"] = "kmeans"
            sil_calc = _silhouette(
                X, labels, sample_size=silhouette_sample_size, seed_for_sample=int(node_seed) + 99991
            )
            diag["silhouette"] = float(sil_calc)
        elif density_driven_k:
            remaining_levels = max(1, L - level)
            K_star = (float(n) / max(1.0, float(target_items_per_leaf))) ** (1.0 / float(remaining_levels))
            max_allowed = min(int(k_cap), n - 1, int(base))
            if max_allowed < 2:
                leaf_counts[level] += 1
                return
            K_star_clamped = int(np.clip(round(K_star), 2, max_allowed))
            K_min = max(2, K_star_clamped - int(k_window_delta))
            K_max = min(max_allowed, K_star_clamped + int(k_window_delta))

            K, labels, diag = _pick_best_k(
                X, item_ids, K_min=K_min, K_max=K_max, seed_local=int(node_seed)
            )
        else:
            K_max = min(int(base), n - 1)  # silhouette needs K < n
            min_k_eff = _capacity_lower_bound_k(n, level)
            if K_max < max(2, min_k_eff):
                leaf_counts[level] += 1
                return

            K, labels, diag = _pick_best_k(
                X, item_ids, K_min=max(2, min_k_eff), K_max=K_max, seed_local=int(node_seed)
            )

        if K <= 1 or labels is None or diag is None:
            leaf_counts[level] += 1
            if stop_recursion_if_no_split:
                return
            _recurse(level + 1, item_ids, node_seed=int(node_seed) + 101)
            return

        if print_each_clustering:
            sil_val = float(diag.get("silhouette", float("nan")))
            sil_str = f"{sil_val:.5f}" if np.isfinite(sil_val) else "nan"
            extra = ""
            if min_share_prob is not None:
                ok = diag["min_share_parent"] + 1e-15 >= float(min_share_prob)
                extra = f" | min_share_req={float(min_share_prob):.6f} | ok={ok}"
            print(
                f"{'  ' * int(level)}[split] lvl={level} n={n} K={int(K)} "
                f"| method={diag.get('assign_method', 'kmeans')} | sil={sil_str} "
                f"| min_prob={float(diag['min_prob_global']):.8f} "
                f"| min_share_parent={float(diag['min_share_parent']):.6f} "
                f"| max_share_parent={float(diag['max_share_parent']):.6f} "
                f"| parent_prob={float(diag['parent_mass_global']):.8f}{extra}"
            )

        sil_val = float(diag.get("silhouette", float("nan")))
        if np.isfinite(sil_val):
            split_sils_by_level[int(level)].append(sil_val)
            split_records.append({
                "level": int(level),
                "n": int(n),
                "K": int(K),
                "silhouette": round(float(sil_val), 5),
            })

        codes[level][item_ids] = labels
        split_counts[level] += 1
        chosen_k_hist[level][int(K)] += 1

        for c in range(int(K)):
            child_ids = item_ids[labels == c]
            if child_ids.size == 0:
                continue
            _recurse(level + 1, child_ids, node_seed=int(node_seed) + 131 + 17 * c)

    if adaptive_depth:
        # Variable-depth tree (AST-Ragged)
        max_d = int(max_depth)
        tau_stop = int(max_items_per_leaf)
        suffix_lvl = max_d
        ragged_item2tokseq_ids: dict[int, list[int]] = {}
        leaf_sizes_list: list[int] = []
        depth_hist = defaultdict(int)

        def _recurse_ragged(level: int, item_ids: np.ndarray, current_prefix: list[int], node_seed: int):
            n = int(item_ids.size)
            if n <= 0:
                return

            # Stopping condition: reached small cluster size or hit max_depth ceiling
            if n <= tau_stop or level >= max_d:
                depth_hist[level] += 1
                leaf_sizes_list.append(n)
                for j, item_id in enumerate(sorted(item_ids.tolist())):
                    # disambiguation token from suffix_lvl
                    ragged_item2tokseq_ids[item_id] = list(current_prefix) + [tok_ids[suffix_lvl][j]]
                return

            X = embs[item_ids]
            if k_schedule is not None:
                K_level = k_schedule[level] if level < len(k_schedule) else k_schedule[-1]
                K = min(int(K_level), n - 1)
                if K < 2:
                    depth_hist[level] += 1
                    leaf_sizes_list.append(n)
                    for j, item_id in enumerate(sorted(item_ids.tolist())):
                        ragged_item2tokseq_ids[item_id] = list(current_prefix) + [tok_ids[suffix_lvl][j]]
                    return
                labels = _regular_kmeans_labels(
                    X, K, seed=int(node_seed), max_iter=max_iter_kmeans, n_init=n_init_kmeans
                )
                diag = _split_stats(item_ids, labels)
            else:
                K_max = min(int(base), n - 1)
                K, labels, diag = _pick_best_k(
                    X, item_ids, K_min=2, K_max=K_max, seed_local=int(node_seed)
                )

            if K <= 1 or labels is None or diag is None:
                depth_hist[level] += 1
                leaf_sizes_list.append(n)
                for j, item_id in enumerate(sorted(item_ids.tolist())):
                    ragged_item2tokseq_ids[item_id] = list(current_prefix) + [tok_ids[suffix_lvl][j]]
                return

            sil_val = float(diag.get("silhouette", float("nan")))
            if np.isfinite(sil_val):
                split_sils_by_level[int(level)].append(sil_val)
                split_records.append({
                    "level": int(level),
                    "n": int(n),
                    "K": int(K),
                    "silhouette": round(float(sil_val), 5),
                })

            if print_each_clustering:
                sil_str = f"{sil_val:.5f}" if np.isfinite(sil_val) else "nan"
                print(f"{'  ' * int(level)}[ragged-split] lvl={level} n={n} K={int(K)} | sil={sil_str}")

            for c in range(int(K)):
                child_ids = item_ids[labels == c]
                if child_ids.size == 0:
                    continue
                next_tok = tok_ids[level][c]
                _recurse_ragged(level + 1, child_ids, current_prefix + [next_tok], int(node_seed) + 131 + 17 * c)

        _recurse_ragged(0, np.arange(N, dtype=np.int32), [], int(seed))

        if print_stats:
            lengths = [len(seq) for seq in ragged_item2tokseq_ids.values()]
            print(f"[AST-Ragged] total_items={len(ragged_item2tokseq_ids)} | total_leaves={len(leaf_sizes_list)}")
            print(f"[AST-Ragged] depth_distribution (leaves per depth): {dict(sorted(depth_hist.items()))}")
            print(f"[AST-Ragged] SID length min={min(lengths)}, median={np.median(lengths)}, max={max(lengths)}")

        if return_tree_stats:
            return ragged_item2tokseq_ids, _build_tree_stats()
        return ragged_item2tokseq_ids

    _recurse(0, np.arange(N, dtype=np.int32), node_seed=int(seed))

    # ------------------------------------------------------------------
    # 4) group items by prefix
    # ------------------------------------------------------------------
    prefix2items: dict[tuple, list[int]] = defaultdict(list)
    for item_id in range(N):
        prefix2items[tuple(tok_ids[lvl][int(codes[lvl][item_id])] for lvl in range(L))].append(item_id)

    bucket_sizes = [len(v) for v in prefix2items.values()]
    max_bucket = max(bucket_sizes) if bucket_sizes else 0
    if max_bucket > base:
        raise ValueError(
            f"Collision suffix needs base >= max_prefix_bucket_size, but max_bucket={max_bucket} > base={base}. "
            f"Try a larger base or m."
        )

    if print_stats:
        n_coll = sum(s for s in bucket_sizes if s > 1)
        print(
            f"[prefix-collision] colliding_buckets={sum(1 for s in bucket_sizes if s > 1)} | "
            f"items_in_collisions={n_coll} ({n_coll / N:.2%}) | max_bucket={max_bucket}"
        )
        for lvl in range(L):
            if node_counts[lvl] == 0:
                print(f"[lvl={lvl}] nodes=0")
            elif split_counts[lvl] == 0:
                print(f"[lvl={lvl}] nodes={int(node_counts[lvl])} | splits=0 | leaves={int(leaf_counts[lvl])}")
            else:
                top = sorted(chosen_k_hist[lvl].items(), key=lambda x: (-x[1], x[0]))[:8]
                print(
                    f"[lvl={lvl}] nodes={int(node_counts[lvl])} | splits={int(split_counts[lvl])} | "
                    f"leaves={int(leaf_counts[lvl])} | topK: " + ", ".join(f"K={k}:{c}" for k, c in top)
                )

    # ------------------------------------------------------------------
    # 5) last token: deterministic index of the item inside its prefix bucket
    # ------------------------------------------------------------------
    suffix_lvl = m - 1
    item2tokseq_ids: dict[int, list[int]] = {}
    for prefix, items in prefix2items.items():
        for j, item_id in enumerate(sorted(items)):
            item2tokseq_ids[item_id] = list(prefix) + [tok_ids[suffix_lvl][j]]

    # ------------------------------------------------------------------
    # 6) verify uniqueness
    # ------------------------------------------------------------------
    if print_stats:
        seq2items: dict[tuple, list[int]] = defaultdict(list)
        for item_id, seq in item2tokseq_ids.items():
            seq2items[tuple(seq)].append(item_id)
        collisions = {seq: ids for seq, ids in seq2items.items() if len(ids) > 1}
        if collisions:
            print(f"[sid-collision AFTER suffix] {len(collisions)} collide")
            for seq, ids in list(collisions.items())[:show_examples]:
                print(f"  seq={seq} -> items={ids[:20]}")
        else:
            print("[sid-collision AFTER suffix] none")

    if return_tree_stats:
        return item2tokseq_ids, _build_tree_stats()
    return item2tokseq_ids
