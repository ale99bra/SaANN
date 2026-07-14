# __init__.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

from . import activation_functions
from . import gradients
from . import initiations
from . import layers
from . import losses
from . import models
from . import processing
from . import metrics
from . import generation
from . import tokenizer
from . import training

# Lazy-load backend to avoid CuPy import on CPU-only systems
def __getattr__(name):
    if name == "backend":
        from . import backend
        return backend
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")