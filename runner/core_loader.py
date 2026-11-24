"""Load the upstream MTJ neuron + network classes WITHOUT running its script.

We slice the upstream file between its section markers (hyper-parameters / neuron /
network classes), drop the dataset and training sections, and exec the rest. The
upstream source is therefore used verbatim; we only override module constants
(THRESHOLD, NEG_THRESHOLD, TAU_LEAK_PS) before the model is constructed.
"""
import re
import types
from pathlib import Path


def load_core(upstream_py):
    path = Path(upstream_py)
    lines = path.read_text(encoding="utf-8").splitlines()

    def idx(marker):
        for i, line in enumerate(lines):
            if marker in line:
                return i
        raise RuntimeError(f"marker not found in {path}: {marker!r}")

    i_dataset = idx("# MNIST Dataset") - 1            # '# ====' line above the marker
    i_core = idx("# Asymmetric ternary OUTPUT SPIKE") - 1
    i_init = idx("# Initialize Model") - 1
    keep = lines[:i_dataset] + lines[i_core:i_init]
    keep = [ln for ln in keep
            if not re.match(r"\s*(from torchvision|import torchvision|from snntorch|import snntorch)", ln)]
    mod = types.ModuleType("mtj_core")
    mod.__file__ = str(path)
    exec(compile("\n".join(keep), f"{path}[sliced]", "exec"), mod.__dict__)
    return mod
