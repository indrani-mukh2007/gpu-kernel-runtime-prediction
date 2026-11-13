import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


FEAT_GROUPS = {
    "FLOPs": ["flops", "log_flops", "t_compute_us"],
    "Memory Footprint": ["memory_bytes", "log_memory_bytes", "memory_mb", "t_memory_us"],
    "Roofline Estimate": ["roofline_us", "log_roofline_us"],
    "Arithmetic Intensity": ["arithmetic_intensity", "log_ai"],
    "Input/Output Shapes": ["batch_size", "dim_m", "dim_n", "dim_k", "is_aligned"],
    "Data Type (dtype)": ["dtype_bytes"],
}

FEAT_COLS = [col for grp in FEAT_GROUPS.values() for col in grp]


def calc_mape(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp) / np.maximum(yt, 1e-9)) * 100.0)


def get_features(df, gpu="T4"):
    g = gpu.lower()
    x = pd.DataFrame(index=df.index)

    x["flops"] = df["flops"].values
    x["log_flops"] = np.log1p(df["flops"].values)
    x["t_compute_us"] = df[f"t_compute_{g}_us"].values

    x["memory_bytes"] = df["memory_bytes"].values
    x["log_memory_bytes"] = np.log1p(df["memory_bytes"].values)
    x["memory_mb"] = df["memory_mb"].values
    x["t_memory_us"] = df[f"t_memory_{g}_us"].values

    x["roofline_us"] = df[f"roofline_{g}_us"].values
    x["log_roofline_us"] = np.log1p(df[f"roofline_{g}_us"].values)

    x["arithmetic_intensity"] = df["arithmetic_intensity"].values
    x["log_ai"] = np.log1p(df["arithmetic_intensity"].values)

    x["batch_size"] = df["batch_size"].values
    x["dim_m"] = df["dim_m"].values
    x["dim_n"] = df["dim_n"].values
    x["dim_k"] = df["dim_k"].values
    x["is_aligned"] = df["is_aligned"].values

    x["dtype_bytes"] = df["dtype_bytes"].values

    return x[FEAT_COLS]


class GlobalModel:
    def __init__(self, seed=42):
        self.cols = [
            "flops", "log_flops", "t_compute_us",
            "arithmetic_intensity", "roofline_us", "log_memory_bytes",
            "batch_size", "dim_m", "dim_n", "dim_k", "is_aligned", "dtype_bytes",
        ]
        self.reg = GradientBoostingRegressor(
            n_estimators=82,
            max_depth=3,
            learning_rate=0.058,
            max_features=0.52,
            subsample=0.85,
            random_state=seed,
        )

    def fit(self, x, y_us):
        c = [col for col in self.cols if col in x.columns]
        high_ai = x["arithmetic_intensity"].values > 10.0
        w = np.where(high_ai, 2.5, 0.35)
        self.reg.fit(x[c], np.log(y_us), sample_weight=w)
        return self

    def predict(self, x):
        c = [col for col in self.cols if col in x.columns]
        return np.exp(self.reg.predict(x[c]))


class ClassModel:
    def __init__(self, n_trees=120, depth=4, lr=0.05, seed=42):
        self.regs = {
            "compute_bound": GradientBoostingRegressor(
                n_estimators=n_trees,
                max_depth=depth,
                learning_rate=lr,
                subsample=0.85,
                min_samples_leaf=5,
                random_state=seed,
            ),
            "memory_bound": GradientBoostingRegressor(
                n_estimators=n_trees,
                max_depth=depth,
                learning_rate=lr,
                subsample=0.85,
                min_samples_leaf=5,
                random_state=seed,
            ),
        }

    def fit(self, x, y_us, classes):
        for cls_name, reg in self.regs.items():
            m = (classes == cls_name).values
            reg.fit(x.loc[m], np.log(y_us[m]))
        return self

    def predict(self, x, classes):
        out = np.zeros(len(x), dtype=float)
        for cls_name, reg in self.regs.items():
            m = (classes == cls_name).values
            if np.any(m):
                out[m] = np.exp(reg.predict(x.loc[m]))
        return out


def eval_t4_models(df):
    x = get_features(df, gpu="T4")
    y = df["runtime_t4_us"].values
    classes = df["op_class"]

    tr = (df["split"] == "train").values
    te = (df["split"] == "test").values
    c_te = te & (classes == "compute_bound").values
    m_te = te & (classes == "memory_bound").values

    p_roof = df["roofline_t4_us"].values

    glob_mod = GlobalModel(seed=42)
    glob_mod.fit(x.loc[tr], y[tr])
    p_glob = np.zeros_like(y)
    p_glob[te] = glob_mod.predict(x.loc[te])

    cls_mod = ClassModel(seed=42)
    cls_mod.fit(x.loc[tr], y[tr], classes.loc[tr])
    p_cls = np.zeros_like(y)
    p_cls[te] = cls_mod.predict(x.loc[te], classes.loc[te])

    rows = []
    for label, preds in [
        ("Class-Specific XGBoost/GBDT", p_cls),
        ("Global XGBoost/GBDT", p_glob),
        ("Analytical Roofline Baseline", p_roof),
    ]:
        rows.append({
            "Model": label,
            "Overall_MAPE_pct": round(calc_mape(y[te], preds[te]), 2),
            "Compute_Bound_MAPE_pct": round(calc_mape(y[c_te], preds[c_te]), 2),
            "Memory_Bound_MAPE_pct": round(calc_mape(y[m_te], preds[m_te]), 2),
        })

    return pd.DataFrame(rows), cls_mod, glob_mod
