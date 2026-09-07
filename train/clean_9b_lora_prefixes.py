"""
为已编译的 9B LoRA 检查点生成可直接加载的副本
"""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(r"F:\CODE\competition\MangoTV\.Checkpoint\9b_V1")
DEFAULT_STEPS = (46800, 57200, 62400, 67600)
PREFIX = "_orig_mod."


def first_key(path: Path) -> str:
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        return next(iter(checkpoint.keys()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("steps", nargs="*", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()
    root = args.root
    steps = args.steps
    for step in steps:
        path = root / f"step-{step}.safetensors"
        raw_path = root / f"step-{step}.raw.safetensors"
        if raw_path.exists():
            raise FileExistsError(f"Refusing to overwrite raw backup: {raw_path}")

        state = load_file(str(path), device="cpu")
        if not state or not all(key.startswith(PREFIX) for key in state):
            raise ValueError(f"Unexpected checkpoint keys in {path}: {next(iter(state), '<empty>')}")
        cleaned = {key.removeprefix(PREFIX): tensor for key, tensor in state.items()}
        if len(cleaned) != len(state):
            raise ValueError(f"Key collision while cleaning {path}")

        temp_path = path.with_suffix(".cleaning.safetensors")
        save_file(cleaned, str(temp_path))
        reloaded = load_file(str(temp_path), device="cpu")
        if set(reloaded) != set(cleaned):
            raise ValueError(f"Key mismatch after writing {temp_path}")
        if not all(torch.equal(reloaded[key], tensor) for key, tensor in cleaned.items()):
            raise ValueError(f"Tensor mismatch after writing {temp_path}")
        if first_key(temp_path).startswith(PREFIX):
            raise ValueError(f"Prefix removal failed for {temp_path}")

        path.replace(raw_path)
        temp_path.replace(path)
        print(
            f"step-{step}: {len(cleaned)} tensors; raw={raw_path.name}; "
            f"cleaned_first_key={first_key(path)}"
        )


if __name__ == "__main__":
    main()
