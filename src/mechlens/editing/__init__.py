"""MechLens editing modules.

Provides ROME and MEMIT weight editing as BASELINE comparison methods.
Only supported for Qwen2.5 and Pythia-1.4B (NOT Llama per R4).
"""

from mechlens.editing import memit, rome
from mechlens.editing.memit import edit as memit_edit
from mechlens.editing.memit import verify_model_support
from mechlens.editing.rome import edit as rome_edit

__all__ = [
    # Submodules
    "rome",
    "memit",
    # Main functions
    "rome_edit",
    "memit_edit",
    "verify_model_support",
]
