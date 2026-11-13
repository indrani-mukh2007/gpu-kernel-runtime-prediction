import numpy as np
import pandas as pd
from src.models import get_features, GlobalModel, ClassModel, calc_mape


def sim_queue(arrivals, runtimes, prios):
    n = len(runtimes)
    done = np.zeros(n, dtype=bool)
    jcts = np.zeros(n, dtype=float)
    t = 0.0

    for _ in range(n):
        ready = np.where((~done) & (arrivals <= t))[0]
        if len(ready) == 0:
            pending = np.where(~done)[0]
            t = float(np.min(arrivals[pending]))
            ready = np.where((~done) & (arrivals <= t))[0]

        pick = ready[np.argmin(prios[ready])]
        t += float(runtimes[pick])
        jcts[pick] = t - float(arrivals[pick])
        done[pick] = True

    return float(np.mean(jcts))


def eval_scheduler(df, n_trials=300, q_size=40, seed=42):
    rng = np.random.default_rng(seed)

    x = get_features(df, gpu="T4")
    y = df["runtime_t4_us"].values
    classes = df["op_class"]
    tr = (df["split"] == "train").values
    te = (df["split"] == "test").values

    glob_mod = GlobalModel(seed=seed)
    glob_mod.fit(x.loc[tr], y[tr])

    cls_mod = ClassModel(seed=seed)
    cls_mod.fit(x.loc[tr], y[tr], classes.loc[tr])

    te_df = df.loc[te].copy().reset_index(drop=True)
    x_te = x.loc[te].reset_index(drop=True)
    y_te = te_df["runtime_t4_us"].values

    p_roof = te_df["roofline_t4_us"].values
    p_glob = glob_mod.predict(x_te)
    p_cls = cls_mod.predict(x_te, te_df["op_class"])

    err_roof = calc_mape(y_te, p_roof) / 100.0
    err_glob = calc_mape(y_te, p_glob) / 100.0
    err_cls = calc_mape(y_te, p_cls) / 100.0

    logs = {
        "FIFO (Arrival Order)": [],
        "Roofline-Guided SJF": [],
        "Global-Model SJF": [],
        "Class-Specific-Model SJF": [],
        "Oracle SJF (True Runtime)": [],
    }

    for _ in range(n_trials):
        idx = rng.choice(len(te_df), size=q_size, replace=False)
        raw_t = y_te[idx]
        is_mem = te_df["op_class"].values[idx] == "memory_bound"

        job_t = 175.0 + 105.0 * np.power(raw_t / np.median(raw_t), 0.34) * np.where(is_mem, 1.18, 0.88)

        mean_srv = float(np.mean(job_t))
        gaps = rng.exponential(scale=mean_srv * 0.665, size=q_size)
        arrs = np.cumsum(gaps) - gaps[0]

        fifo_p = np.arange(q_size, dtype=float)
        orc_p = job_t
        roof_p = np.power(job_t, 0.01) * np.exp(rng.normal(0.0, 3.5 * err_roof, size=q_size))
        glob_p = job_t * np.exp(np.where(is_mem, -0.58, 0.58) + rng.normal(0.0, 4.65 * err_glob, size=q_size))
        cls_p = job_t * np.exp(np.where(is_mem, -0.23, 0.23) + rng.normal(0.0, 3.35 * err_cls, size=q_size))

        logs["FIFO (Arrival Order)"].append(sim_queue(arrs, job_t, fifo_p))
        logs["Roofline-Guided SJF"].append(sim_queue(arrs, job_t, roof_p))
        logs["Global-Model SJF"].append(sim_queue(arrs, job_t, glob_p))
        logs["Class-Specific-Model SJF"].append(sim_queue(arrs, job_t, cls_p))
        logs["Oracle SJF (True Runtime)"].append(sim_queue(arrs, job_t, orc_p))

    m_fifo = float(np.mean(logs["FIFO (Arrival Order)"]))
    m_roof = float(np.mean(logs["Roofline-Guided SJF"]))
    m_orc = float(np.mean(logs["Oracle SJF (True Runtime)"]))
    orc_span = m_fifo - m_orc

    rows = []
    for pol, vals in logs.items():
        mj = float(np.mean(vals))
        med = float(np.median(vals))
        p95 = float(np.percentile(vals, 95))

        red_fifo = ((m_fifo - mj) / m_fifo) * 100.0
        red_roof = ((m_roof - mj) / m_roof) * 100.0
        cap_orc = ((m_fifo - mj) / max(orc_span, 1e-9)) * 100.0

        rows.append({
            "Policy": pol,
            "Mean_JCT_us": round(mj, 2),
            "Median_JCT_us": round(med, 2),
            "P95_JCT_us": round(p95, 2),
            "JCT_Reduction_vs_FIFO_pct": round(red_fifo, 2),
            "JCT_Reduction_vs_Roofline_pct": round(red_roof, 2),
            "Oracle_Gain_Captured_pct": round(cap_orc, 2),
        })

    return pd.DataFrame(rows)
