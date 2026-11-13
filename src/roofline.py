import numpy as np


HW_SPECS = {
    "T4": {
        "fp32_tflops": 8.1,
        "fp16_tflops": 65.0,
        "bw_gbps": 320.0,
        "launch_us": 6.5,
        "l2_mb": 4.0,
    },
    "P100": {
        "fp32_tflops": 9.3,
        "fp16_tflops": 18.7,
        "bw_gbps": 732.0,
        "launch_us": 8.2,
        "l2_mb": 4.0,
    },
}


def calc_roofline(flops, mem_bytes, dtype_bytes, gpu):
    spec = HW_SPECS[gpu]
    peak_tflops = np.where(dtype_bytes == 2, spec["fp16_tflops"], spec["fp32_tflops"])
    bw = spec["bw_gbps"]

    t_comp = (flops / (peak_tflops * 1e12)) * 1e6
    t_mem = (mem_bytes / (bw * 1e9)) * 1e6
    roof_us = np.maximum(t_comp, t_mem)

    ai = flops / np.maximum(mem_bytes, 1.0)
    ridge = (peak_tflops * 1e3) / bw

    return {
        "t_compute_us": t_comp,
        "t_memory_us": t_mem,
        "roofline_us": roof_us,
        "arithmetic_intensity": ai,
        "ridge_point": ridge,
    }
