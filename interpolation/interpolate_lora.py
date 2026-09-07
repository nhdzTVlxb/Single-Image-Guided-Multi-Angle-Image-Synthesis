"""对两个同结构 LoRA safetensors 检查点做离线线性插值。

最终 L75 权重的等价生成命令：

    python interpolate_lora.py \
        --checkpoint-a e50_resume_step-31200.safetensors \
        --checkpoint-b e65_resume_step-62400.safetensors \
        --alpha 0.75 \
        --output e50_e65_l75.safetensors

输出定义为：output = (1 - alpha) * checkpoint_a + alpha * checkpoint_b。
本脚本只处理模型权重，不参与推理，也不会把两个检查点同时加载到提交模型中。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="checkpoint-b 的权重；checkpoint-a 的权重为 1-alpha",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def inspect_checkpoint(path: Path) -> dict[str, tuple[torch.Size, torch.dtype]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt", device="cpu") as archive:
        return {
            key: (
                torch.Size(archive.get_slice(key).get_shape()),
                archive.get_tensor(key).dtype,
            )
            for key in archive.keys()
        }


def validate_inputs(
    checkpoint_a: Path,
    checkpoint_b: Path,
    output: Path,
    alpha: float,
) -> dict[str, tuple[torch.Size, torch.dtype]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须位于 [0, 1]，当前为 {alpha}")

    a_resolved = checkpoint_a.resolve()
    b_resolved = checkpoint_b.resolve()
    output_resolved = output.resolve()
    if a_resolved == b_resolved:
        raise ValueError("两个输入检查点不能是同一个文件")
    if output_resolved in {a_resolved, b_resolved}:
        raise ValueError("输出路径不能覆盖输入检查点")

    spec_a = inspect_checkpoint(checkpoint_a)
    spec_b = inspect_checkpoint(checkpoint_b)
    if set(spec_a) != set(spec_b):
        only_a = sorted(set(spec_a) - set(spec_b))
        only_b = sorted(set(spec_b) - set(spec_a))
        raise ValueError(
            "检查点参数键不一致："
            f"only_a={only_a[:5]}，only_b={only_b[:5]}"
        )

    for key in spec_a:
        if spec_a[key] != spec_b[key]:
            raise ValueError(
                f"张量结构不一致：{key}: a={spec_a[key]}, b={spec_b[key]}"
            )
    return spec_a


def interpolate_tensor(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    if a.is_floating_point():
        # LoRA 通常为 BF16/FP16。先转为 FP32 插值，再转回原 dtype，
        # 与本方案实际生成 L75 权重的计算方式一致。
        return (a.float() * (1.0 - alpha) + b.float() * alpha).to(a.dtype)

    # 非浮点状态不适合做数值插值；若两侧完全相同则原样保留。
    if not torch.equal(a, b):
        raise ValueError(f"无法插值不相等的非浮点张量，dtype={a.dtype}")
    return a.clone()


def verify_output(
    output: Path,
    expected: dict[str, torch.Tensor],
    expected_spec: dict[str, tuple[torch.Size, torch.dtype]],
) -> None:
    with safe_open(output, framework="pt", device="cpu") as archive:
        if set(archive.keys()) != set(expected_spec):
            raise RuntimeError("输出文件的参数键与输入不一致")
        for key, (shape, dtype) in expected_spec.items():
            tensor = archive.get_tensor(key)
            if tensor.shape != shape or tensor.dtype != dtype:
                raise RuntimeError(
                    f"输出张量结构异常：{key}: "
                    f"shape={tensor.shape}, dtype={tensor.dtype}"
                )
            if not torch.equal(tensor, expected[key]):
                raise RuntimeError(f"输出文件回读校验失败：{key}")


def main() -> None:
    args = parse_args()
    checkpoint_a = args.checkpoint_a.resolve()
    checkpoint_b = args.checkpoint_b.resolve()
    output = args.output.resolve()
    alpha = float(args.alpha)

    spec = validate_inputs(checkpoint_a, checkpoint_b, output, alpha)
    state_a = load_file(checkpoint_a, device="cpu")
    state_b = load_file(checkpoint_b, device="cpu")
    interpolated = {
        key: interpolate_tensor(state_a[key], state_b[key], alpha)
        for key in sorted(spec)
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    metadata = {
        "format": "pt",
        "interpolation": "direct_lora_parameter",
        "formula": "output=(1-alpha)*checkpoint_a+alpha*checkpoint_b",
        "alpha": f"{alpha:.12g}",
        "checkpoint_a_weight": f"{1.0 - alpha:.12g}",
        "checkpoint_b_weight": f"{alpha:.12g}",
        "checkpoint_a": checkpoint_a.name,
        "checkpoint_b": checkpoint_b.name,
    }

    try:
        save_file(interpolated, temporary, metadata=metadata)
        verify_output(temporary, interpolated, spec)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"已生成：{output}")
    print(f"张量数：{len(interpolated)}")
    print(f"权重：A={1.0 - alpha:.6f}, B={alpha:.6f}")


if __name__ == "__main__":
    main()
