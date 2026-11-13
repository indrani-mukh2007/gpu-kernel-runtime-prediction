import copy
import numpy as np
import pandas as pd
from src.models import get_features, ClassModel, calc_mape


def eval_transfer(df, seed=42):
    rng = np.random.default_rng(seed)

    x_t4 = get_features(df, gpu="T4")
    y_t4 = df["runtime_t4_us"].values

    x_p100 = get_features(df, gpu="P100")
    y_p100 = df["runtime_p100_us"].values

    classes = df["op_class"]
    tr = (df["split"] == "train").values
    te = (df["split"] == "test").values
    c_te = te & (classes == "compute_bound").values
    m_te = te & (classes == "memory_bound").values

    src_mod = ClassModel(n_trees=100, depth=4, lr=0.05, seed=seed)
    src_mod.fit(x_t4.loc[tr], y_t4[tr], classes.loc[tr])

    zs_p = src_mod.predict(x_p100.loc[te], classes.loc[te])
    zs_all = calc_mape(y_p100[te], zs_p)
    zs_comp = calc_mape(y_p100[c_te], zs_p[(classes.loc[te] == "compute_bound").values])
    zs_mem = calc_mape(y_p100[m_te], zs_p[(classes.loc[te] == "memory_bound").values])

    fracs = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    add_trees = {0.01: 14, 0.02: 20, 0.05: 29, 0.10: 41, 0.25: 62, 0.50: 85, 1.00: 115}

    rows = [
        {
            "Transfer_Mode": "Zero-Shot (0% P100 Data)",
            "P100_Data_Fraction_pct": 0.0,
            "P100_Samples_Used": 0,
            "Overall_MAPE_pct": round(zs_all, 2),
            "Compute_Bound_MAPE_pct": round(zs_comp, 2),
            "Memory_Bound_MAPE_pct": round(zs_mem, 2),
        }
    ]

    for f in fracs:
        ft_p = np.zeros(np.sum(te), dtype=float)
        n_used = 0

        for cls_name in ["compute_bound", "memory_bound"]:
            cls_idx = np.where(tr & (classes == cls_name).values)[0]
            k = max(4, int(round(len(cls_idx) * f)))
            n_used += k
            sub_idx = rng.choice(cls_idx, size=k, replace=False)

            reg = copy.deepcopy(src_mod.regs[cls_name])
            extra = add_trees.get(f, 40)
            reg.set_params(
                warm_start=True,
                n_estimators=reg.n_estimators + extra,
                learning_rate=0.042,
            )

            reg.fit(x_p100.iloc[sub_idx], np.log(y_p100[sub_idx]))

            loc_mask = (classes.loc[te] == cls_name).values
            ft_p[loc_mask] = np.exp(reg.predict(x_p100.loc[te].loc[loc_mask]))

        m_all = calc_mape(y_p100[te], ft_p)
        m_comp = calc_mape(y_p100[c_te], ft_p[(classes.loc[te] == "compute_bound").values])
        m_mem = calc_mape(y_p100[m_te], ft_p[(classes.loc[te] == "memory_bound").values])

        rows.append({
            "Transfer_Mode": f"Warm-Start Fine-Tuned ({int(f * 100)}% P100 Data)",
            "P100_Data_Fraction_pct": round(f * 100.0, 1),
            "P100_Samples_Used": n_used,
            "Overall_MAPE_pct": round(m_all, 2),
            "Compute_Bound_MAPE_pct": round(m_comp, 2),
            "Memory_Bound_MAPE_pct": round(m_mem, 2),
        })

    return pd.DataFrame(rows)
