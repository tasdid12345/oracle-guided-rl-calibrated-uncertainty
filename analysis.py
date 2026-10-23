"""Adaptive, reliability-aware critic-uncertainty oracle guidance vs fixed-threshold (CCGE-style), tabular gridworld."""
import json, os, sys
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

N, A, K = 12, 4, 5                  # grid size, actions, critics in the ensemble
EPS, MAXT, GAMMA, LR, EVAL_EVERY = 150, 150, 0.95, 0.3, 5
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EP_AXIS = np.arange(EVAL_EVERY, EPS + 1, EVAL_EVERY)


def make_env(seed=42):
    """Walled 12x12 grid, goal bottom-right, plus a fixed imperfect oracle (good in some regions, poor in others)."""
    rng = np.random.default_rng(seed)
    wall = np.zeros((N, N), bool)
    wall[1:9, 4] = wall[7, 6:11] = wall[0:5, 8] = wall[9, 1:3] = True
    S = N * N
    nxt = np.zeros((S, A), int)
    for r in range(N):
        for c in range(N):
            for a, (dr, dc) in enumerate(MOVES):
                r2, c2 = r + dr, c + dc
                if not (0 <= r2 < N and 0 <= c2 < N) or wall[r2, c2]:
                    r2, c2 = r, c
                nxt[r * N + c, a] = r2 * N + c2
    goal = S - 1
    dist = np.full(S, -1); dist[goal] = 0
    q = deque([goal])
    while q:                         # BFS distance-to-goal (moves are symmetric)
        s = q.popleft()
        for s2 in nxt[s]:
            if dist[s2] < 0 and not wall[s2 // N, s2 % N]:
                dist[s2] = dist[s] + 1; q.append(s2)
    free = np.where(dist > 0)[0]
    region = (np.arange(S) // N // 4) * 3 + (np.arange(S) % N) // 4   # 9 regions of 4x4
    bad = rng.choice(8, 4, replace=False)                              # goal region stays good
    oracle = np.zeros(S, int)
    for s in free:
        d = dist[nxt[s]]
        if region[s] in bad:         # poor region: mostly picks the worst action
            oracle[s] = d.argmax() if rng.random() < 0.8 else rng.integers(A)
        else:                        # good region: mostly optimal
            oracle[s] = rng.choice(np.flatnonzero(d == dist[s] - 1)) if rng.random() < 0.9 else rng.integers(A)
    opt_frac = float(np.mean([dist[nxt[s, oracle[s]]] == dist[s] - 1 for s in free]))
    return dict(nxt=nxt, goal=goal, free=free, region=region, oracle=oracle, opt_frac=opt_frac)


def evaluate(Q, env, cells):
    """Greedy (no oracle, no exploration) success rate from fixed start cells."""
    pol = Q.mean(0).argmax(1)
    s, hit = cells.copy(), np.zeros(len(cells), bool)
    for _ in range(MAXT):
        s = env["nxt"][s, pol[s]]
        hit |= s == env["goal"]
    return hit.mean()


def run(env, method, param, seed, train_cells, eval_cells):
    """One training run. method 'baseline': defer if u > fixed threshold `param`.
    'improved': defer if u > running `param`-quantile of u AND region reliability >= 0.5."""
    rng = np.random.default_rng(seed)
    nxt, goal, oracle, region = env["nxt"], env["goal"], env["oracle"], env["region"]
    Q = rng.normal(0, 0.05, (K, N * N, A))      # ensemble of tabular critics (random init)
    ra, rb = np.full(9, 2.0), np.full(9, 1.0)   # Beta counts of oracle reliability per region
    buf, thr = deque(maxlen=1000), (param if method == "baseline" else np.inf)
    curve, n_or, n_steps = [], 0, 0
    for ep in range(1, EPS + 1):
        eps = max(0.1, 1 - 0.9 * ep / 100)       # shared exploration schedule
        s, used, done = rng.choice(train_cells), [], False
        for t in range(MAXT):
            q = Q[:, s]; qm = q.mean(0); u = q.std(0).mean()   # ensemble disagreement
            g, ao, reg = int(qm.argmax()), oracle[s], region[s]
            if method == "improved":
                buf.append(u)
                if len(buf) >= 20 and n_steps % 20 == 0:
                    thr = np.quantile(buf, param)             # adaptive threshold
                if len(buf) >= 20 and u <= thr:               # critic confident: is oracle action ~ greedy?
                    if qm[ao] >= qm[g] - 0.02: ra[reg] += 0.25
                    else: rb[reg] += 0.25
                use = u > thr and ra[reg] / (ra[reg] + rb[reg]) >= 0.5
            else:
                use = u > thr
            if use:
                a = ao; n_or += 1; used.append(reg)
            elif rng.random() < eps:
                a = rng.integers(A)
            else:
                a = g
            s2 = nxt[s, a]; done = s2 == goal
            tgt = np.full(K, 1.0) if done else GAMMA * Q[:, s2].max(1)
            m = rng.random(K) < 0.5                            # bootstrap mask per critic
            Q[m, s, a] += LR * (tgt[m] - Q[m, s, a])
            s = s2; n_steps += 1
            if done: break
        if method == "improved":                               # realised-return evidence for oracle steps
            for reg in used:
                if done: ra[reg] += 0.5
                else: rb[reg] += 0.5
        if ep % EVAL_EVERY == 0:
            curve.append(evaluate(Q, env, eval_cells))
    return np.array(curve), n_or / max(n_steps, 1)


def main():
    env = make_env(42)
    train_c, test_c = train_test_split(env["free"], test_size=0.25, random_state=42)  # start-cell split
    par = Parallel(n_jobs=-1)
    # small grid per method, tuned on separate seeds using TRAIN start cells only
    grids = {"baseline": [0.01, 0.02, 0.04, 0.08], "improved": [0.5, 0.7, 0.9]}
    best = {}
    for meth, grid in grids.items():
        res = par(delayed(run)(env, meth, p, sd, train_c, train_c) for p in grid for sd in (9001, 9002))
        sc = [np.mean([res[2 * i + j][0].mean() for j in range(2)]) for i in range(len(grid))]
        best[meth] = grid[int(np.argmax(sc))]
        print(f"tuned {meth}: param={best[meth]} (train-start AUC {max(sc):.3f})", flush=True)
    # final: 3 master seeds (42,43,44) x 10 shared run seeds, evaluated on fixed held-out start cells
    out = {}
    for meth in grids:
        res = par(delayed(run)(env, meth, best[meth], m * 100 + k, train_c, test_c)
                  for m in (42, 43, 44) for k in range(10))
        out[meth] = (np.array([r[0] for r in res]).reshape(3, 10, -1), np.array([r[1] for r in res]).reshape(3, 10))
        print(f"finished {meth}", flush=True)

    def e80(c):
        i = np.argmax(c >= 0.8)
        return float(EP_AXIS[i]) if c[i] >= 0.8 else float(EPS)   # censored at budget

    stats = {"auc": lambda c, f: c.mean(-1), "final": lambda c, f: c[..., -1],
             "episodes_to_80": lambda c, f: np.apply_along_axis(e80, -1, c), "oracle_frac": lambda c, f: f}
    metrics = {}
    for meth in grids:
        for name, fn in stats.items():
            per_master = fn(*out[meth]).mean(1)       # mean over 10 run seeds for each master seed
            key = {"auc": "score"}.get(name, name)
            metrics[f"{meth}_{key}"] = float(per_master.mean())
            metrics[f"{meth}_{key}_std"] = float(per_master.std())
    metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]
    metrics["baseline_param"], metrics["improved_param"] = float(best["baseline"]), float(best["improved"])
    metrics["oracle_optimal_action_fraction"] = env["opt_frac"]
    if not all(np.isfinite(v) for v in metrics.values()):
        sys.exit("non-finite metrics")

    os.makedirs("figures", exist_ok=True)
    for meth, lab in (("baseline", "fixed-threshold (CCGE-style)"), ("improved", "adaptive + reliability")):
        c = out[meth][0].reshape(-1, len(EP_AXIS))
        plt.plot(EP_AXIS, c.mean(0), label=lab)
        plt.fill_between(EP_AXIS, c.mean(0) - c.std(0) / np.sqrt(len(c)), c.mean(0) + c.std(0) / np.sqrt(len(c)), alpha=0.2)
    plt.xlabel("training episode"); plt.ylabel("greedy success rate (held-out starts)"); plt.legend()
    plt.savefig("figures/learning_curves.png", dpi=120, bbox_inches="tight")

    results = {
        "question": "In a simple sparse-reward gridworld with an imperfect oracle, can an adaptively calibrated, oracle-reliability-aware critic-uncertainty rule learn more sample-efficiently than a fixed-threshold critic-uncertainty rule (CCGE-style)?",
        "method": "Seeded 12x12 walled sparse-reward gridworld in numpy with a region-dependent imperfect oracle; tabular Q-learning with 5 bootstrapped critics, same exploration schedule and 150-episode budget for both agents; greedy evaluation every 5 episodes on fixed held-out start cells; 3 master seeds (42,43,44) x 10 shared run seeds.",
        "dataset": {"name": "Simulated sparse-reward gridworld with imperfect oracle", "source_url": "none (simulated in numpy)",
                    "n_rows": int(len(env["free"])), "n_train": int(len(train_c)), "n_test": int(len(test_c)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "mean greedy-evaluation success rate over training (area under learning curve)",
        "higher_is_better": True,
        "baseline": "Fixed-threshold rule: follow the oracle action when ensemble std of Q at the state exceeds a fixed threshold (tuned on train start cells); simplified tabular stand-in for CCGE.",
        "improved": "Threshold is a running quantile of observed uncertainty, and a per-region oracle-reliability estimate (critic value of oracle vs greedy action, plus realised success) gates deferral to the oracle.",
        "metrics": metrics,
        "notes": "Simplified tabular stand-in, not the paper's actor-critic CCGE. Test set = held-out start cells (25% of free cells); hyper-parameters (threshold / quantile) chosen on separate seeds using training start cells only; other constants fixed a priori. Episodes-to-80% is censored at 150 if never reached. Realised-return evidence for reliability is noisy (a failed episode penalises all oracle steps in it). Results depend on the simulated oracle quality and layout.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
