# 检查点插值说明

最终 LoRA 为同一训练轨迹中 E50 与 E65 的逐张量线性插值：

```text
L75 = 0.25 × E50 + 0.75 × E65
```

输入权重位于 `../checkpoints/e50.safetensors` 和 `../checkpoints/e65.safetensors`，输出为 `../checkpoints/e50_e65_l75.safetensors`。三个文件均为 LoRA 权重；E50、E65各有288个同名张量，键、shape和dtype一致。

## 生成方式

`interpolate_lora.py` 对每个同名张量先转 FP32，计算后转回原dtype：

```python
output = (e50.float() * 0.25 + e65.float() * 0.75).to(e50.dtype)
```

脚本会在写入后逐张量回读并校验。示例：

```powershell
python .\interpolate_lora.py `
  --checkpoint-a ..\checkpoints\e50.safetensors `
  --checkpoint-b ..\checkpoints\e65.safetensors `
  --alpha 0.75 `
  --output ..\checkpoints\e50_e65_l75.safetensors
```

该处理在推理前离线完成。最终推理仅加载一个L75文件，不会同时加载E50/E65，也不在推理时动态融合权重。

## 选择依据

插值可在同一训练轨迹的两个状态（本方案中特指Epoch 50和Epoch 65）之间寻找更合适的参数位置，在不重新训练的情况下组合两个检查点的参数状态，获得更稳定、更泛化的图像表现。

其使用主要针对E50到E65之间出现的分数回落现象：在20步、CFG1、seed0的同推理参数下，E55线上得分1.8892，E60得分1.8800，而E50和E65都高于1.89。

为检验插值，以14 steps、CFG1、seed0/42等权TTA进行A榜对照（TTA仅为过程遗留，不是最终配置）：

| LoRA | 总分 | PSNR | SSIM | LPIPS | NRIQA |
| --- | ---: | ---: | ---: | ---: | ---: |
| E50 | 1.909942 | 0.3199 | 0.3647 | 0.4664 | 0.7589 |
| E65 | 1.9098 | 0.3198 | 0.3652 | 0.4650 | 0.7599 |
| L75 | 1.9111 | 0.3201 | 0.3653 | 0.4655 | 0.7602 |

L75相较E50提升约0.001158、相较E65提升约0.0013，因此用于后续版本及最终提交。

