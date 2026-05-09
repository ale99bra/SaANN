# __init__.py
from . import activation_functions
from . import gradients
from . import initiations
from . import layers
from . import losses
from . import models
from . import processing
from . import metrics

# Lazy-load backend to avoid CuPy import on CPU-only systems
def __getattr__(name):
    if name == "backend":
        from . import backend
        return backend
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")