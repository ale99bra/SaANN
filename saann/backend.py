# backend.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cupy")

def safe_convert(obj):
    import cupy as cp
    import numpy as np
    # Convert CuPy → NumPy
    if isinstance(obj, cp.ndarray):
        return cp.asnumpy(obj)

    # Leave NumPy arrays untouched
    if isinstance(obj, np.ndarray):
        return obj

    # Recursively convert lists
    if isinstance(obj, list):
        return [safe_convert(x) for x in obj]

    # Recursively convert tuples
    if isinstance(obj, tuple):
        return tuple(safe_convert(x) for x in obj)

    # Recursively convert dicts
    if isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}

    # DO NOT touch custom objects
    return obj

def cupy_to_numpy(obj):
    if isinstance(obj, xp.ndarray):
        return xp.asnumpy(obj)

    elif isinstance(obj, list):
        return [cupy_to_numpy(x) for x in obj]

    elif isinstance(obj, tuple):
        return tuple(cupy_to_numpy(x) for x in obj)

    elif isinstance(obj, dict):
        return {k: cupy_to_numpy(v) for k, v in obj.items()}

    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            # Only convert if it's actually a CuPy array
            if isinstance(v, xp.ndarray):
                obj.__dict__[k] = xp.asnumpy(v)
            else:
                obj.__dict__[k] = cupy_to_numpy(v)
        return obj

    return obj


def to_numpy(x):
    # CuPy arrays have .get()
    if hasattr(x, "get"):
        return x.get()
    return x

def use_cpu():
    global FORCE_CPU, xp, gpu_available
    FORCE_CPU = True
    import numpy as np
    xp = np
    gpu_available = False

def use_gpu():
    global FORCE_CPU, xp, gpu_available
    FORCE_CPU = False
    import cupy as cp
    xp = cp
    gpu_available = True


def add_cupy_dll_path():
    if sys.platform.startswith("win"):
        try:
            import cupy
            cupy_path = os.path.dirname(cupy.__file__)
            dll_path = os.path.join(cupy_path, "cuda", "lib")

            if dll_path not in os.environ["PATH"]:
                os.environ["PATH"] = dll_path + os.pathsep + os.environ["PATH"]

        except Exception as e:
            print("Failed to add CuPy DLL path:", e)


add_cupy_dll_path()

try:
    import cupy as xp
    dtype = xp.float32
    gpu_available = True

    # Warm up GPU
    a = xp.zeros((10,), dtype=xp.float32)
    b = xp.zeros((10,), dtype=xp.float32)
    res = xp.dot(a, b)
    print("GPU warm-up successful!")

except ImportError as e:
    import numpy as xp
    dtype = xp.float32
    gpu_available = False
    print("GPU failed:", e)