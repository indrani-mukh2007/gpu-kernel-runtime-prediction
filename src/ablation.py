import pandas as pd
from src.models import FEAT_GROUPS, get_features, ClassModel, calc_mape


def run_ablation(df, gpu="T4"):
    x_full = get_features(df, gpu=gpu)
    y = df[f"runtime_{gpu.lower()}_us"].values
    classes = df["op_class"]

    tr = (df["split"] == "train").values
    te = (df["split"] == "test").values
    c_te = te & (classes == "compute_bound").values
    m_te = te & (classes == "memory_bound").values

    base_mod = ClassModel(seed=42)
    base_mod.fit(x_full.loc[tr], y[tr], classes.loc[tr])
    base_p = base_mod.predict(x_full.loc[te], classes.loc[te])

    b_all = calc_mape(y[te], base_p)
    b_comp = calc_mape(y[c_te], base_p[(classes.loc[te] == "compute_bound").values])
    b_mem = calc_mape(y[m_te], base_p[(classes.loc[te] == "memory_bound").values])

    rows = [
        {
            "Ablated_Feature_Group": "None (Full Model)",
            "Overall_MAPE_pct": round(b_all, 2),
            "Compute_Bound_MAPE_pct": round(b_comp, 2),
            "Memory_Bound_MAPE_pct": round(b_mem, 2),
            "Delta_Overall_MAPE_pct": 0.00,
            "Delta_Compute_Bound_MAPE_pct": 0.00,
            "Delta_Memory_Bound_MAPE_pct": 0.00,
        }
    ]

    for grp_name, drop_cols in FEAT_GROUPS.items():
        keep = [c for c in x_full.columns if c not in drop_cols]
        x_sub = x_full[keep]

        mod = ClassModel(seed=42)
        mod.fit(x_sub.loc[tr], y[tr], classes.loc[tr])
        preds = mod.predict(x_sub.loc[te], classes.loc[te])

        m_all = calc_mape(y[te], preds)
        m_comp = calc_mape(y[c_te], preds[(classes.loc[te] == "compute_bound").values])
        m_mem = calc_mape(y[m_te], preds[(classes.loc[te] == "memory_bound").values])

        rows.append({
            "Ablated_Feature_Group": f"- {grp_name}",
            "Overall_MAPE_pct": round(m_all, 2),
            "Compute_Bound_MAPE_pct": round(m_comp, 2),
            "Memory_Bound_MAPE_pct": round(m_mem, 2),
            "Delta_Overall_MAPE_pct": round(m_all - b_all, 2),
            "Delta_Compute_Bound_MAPE_pct": round(m_comp - b_comp, 2),
            "Delta_Memory_Bound_MAPE_pct": round(m_mem - b_mem, 2),
        })

    return pd.DataFrame(rows)
