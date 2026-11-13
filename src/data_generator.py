import numpy as np
import pandas as pd
from src.roofline import calc_roofline


OP_SPECS = [
    {"op_name": "gemm", "op_class": "compute_bound", "split": "train"},
    {"op_name": "bmm", "op_class": "compute_bound", "split": "train"},
    {"op_name": "conv2d_3x3", "op_class": "compute_bound", "split": "train"},
    {"op_name": "conv2d_1x1", "op_class": "compute_bound", "split": "train"},
    {"op_name": "sdpa_attention", "op_class": "compute_bound", "split": "test"},
    {"op_name": "layernorm", "op_class": "memory_bound", "split": "train"},
    {"op_name": "softmax", "op_class": "memory_bound", "split": "train"},
    {"op_name": "elementwise_gelu", "op_class": "memory_bound", "split": "train"},
    {"op_name": "reduce_sum", "op_class": "memory_bound", "split": "train"},
    {"op_name": "rmsnorm", "op_class": "memory_bound", "split": "test"},
]


def make_dataset(n_total=2000, seed=42):
    rng = np.random.default_rng(seed)
    n_per_op = n_total // len(OP_SPECS)
    rows = []

    for spec in OP_SPECS:
        op_name = spec["op_name"]
        op_class = spec["op_class"]
        split = spec["split"]

        for _ in range(n_per_op):
            dtype = rng.choice(["fp32", "fp16"], p=[0.65, 0.35])
            d_bytes = 4 if dtype == "fp32" else 2

            bs = int(rng.choice([1, 2, 4, 8, 16, 32, 64]))
            m = int(rng.integers(128, 2048))
            n = int(rng.integers(128, 2048))
            k = int(rng.integers(64, 1024))
            aligned = int((m % 16 == 0) and (n % 16 == 0))

            vol = float(bs) * float(m) * float(n)

            if op_class == "compute_bound":
                nom_f = 2.0 * vol * float(k) * 0.50
                nom_m = (vol + float(m * k) + float(k * n)) * d_bytes * 0.15

                f_scale = np.exp(rng.uniform(-0.34, 0.36))
                flops = float(nom_f * f_scale)

                m_scale = np.exp(rng.uniform(-0.35, 0.35))
                m_bytes = float(nom_m * m_scale)
            else:
                nom_m = 2.2 * vol * d_bytes * 0.6
                nom_f = 6.0 * vol * 0.6

                m_scale = np.exp(rng.uniform(-0.36, 0.38))
                m_bytes = float(nom_m * m_scale)

                f_scale = np.exp(rng.uniform(-0.40, 0.40))
                flops = float(nom_f * f_scale)

            rows.append({
                "op_name": op_name,
                "op_class": op_class,
                "split": split,
                "batch_size": bs,
                "dim_m": m,
                "dim_n": n,
                "dim_k": k,
                "dtype": dtype,
                "dtype_bytes": d_bytes,
                "is_aligned": aligned,
                "nom_flops": float(nom_f),
                "nom_memory_bytes": float(nom_m),
                "flops": flops,
                "memory_bytes": m_bytes,
                "memory_mb": m_bytes / (1024.0 ** 2),
            })

    df = pd.DataFrame(rows)
    df.insert(0, "config_id", [f"cfg_{i:04d}" for i in range(len(df))])

    n_rows = len(df)
    is_comp = (df["op_class"] == "compute_bound").values
    aligned_arr = df["is_aligned"].values
    mem_mb = df["memory_mb"].values
    flops_g = df["flops"].values / 1e9

    for gpu in ["T4", "P100"]:
        g = gpu.lower()
        nom_res = calc_roofline(
            df["nom_flops"].values,
            df["nom_memory_bytes"].values,
            df["dtype_bytes"].values,
            gpu=gpu,
        )
        true_res = calc_roofline(
            df["flops"].values,
            df["memory_bytes"].values,
            df["dtype_bytes"].values,
            gpu=gpu,
        )

        df[f"t_compute_{g}_us"] = true_res["t_compute_us"]
        df[f"t_memory_{g}_us"] = true_res["t_memory_us"]
        df[f"roofline_{g}_us"] = nom_res["roofline_us"]
        df[f"ai_{g}"] = nom_res["arithmetic_intensity"]

        tc = true_res["t_compute_us"]
        tm = true_res["t_memory_us"]

        if gpu == "T4":
            c_eff = 0.89 - 0.03 * np.exp(-flops_g / 10.0) + 0.02 * aligned_arr
            c_rt = (tc / c_eff) + 0.03 * tm + 2.5

            spill = 1.0 / (1.0 + np.exp(-(mem_mb - 4.0) / 2.5))
            m_eff = 0.762 - 0.05 * spill + 0.02 * aligned_arr
            m_rt = (tm / m_eff) + 0.03 * tc + 3.5
        else:
            fp16_mask = (df["dtype_bytes"] == 2).values
            c_eff = 0.675 - 0.05 * np.exp(-flops_g / 12.0) + 0.03 * aligned_arr - 0.06 * fp16_mask
            c_rt = (tc / c_eff) + 0.08 * tm + 7.5

            spill = 1.0 / (1.0 + np.exp(-(mem_mb - 4.0) / 2.5))
            m_eff = 0.324 - 0.05 * spill + 0.03 * aligned_arr
            m_rt = (tm / m_eff) + 0.08 * tc + 8.5

        base_rt = np.where(is_comp, c_rt, m_rt)
        sig = np.where(df["split"].values == "test", 0.102, 0.075)
        eps = rng.normal(0.0, sig, size=n_rows)
        df[f"runtime_{g}_us"] = base_rt * np.exp(eps)

    df["arithmetic_intensity"] = df["nom_flops"] / np.maximum(df["nom_memory_bytes"], 1.0)
    return df
