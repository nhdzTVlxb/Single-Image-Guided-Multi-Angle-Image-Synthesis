# 训练说明

## 范围与环境

本方案只使用比赛提供的 `dataA_1`，不使用外部数据。训练基座为 `black-forest-labs/FLUX.2-klein-9B`，方式为 SFT LoRA 微调。

训练均在同一云服务商租用的单张 NVIDIA A100-PCIE-40GB 上完成：Ubuntu 22.04.4 LTS、Linux 6.8.0-49-generic、8 vCPU Intel Xeon Gold 6248R、约 62 GiB 内存、NVIDIA driver 550.127.08、CUDA 12.4.1、cuDNN 9.5.1.17、系统 Python 3.10.12。原训练时的精确 PyTorch/依赖版本未完整保留。

## 数据与固定参数

`../data/` 有20个样本目录，每个目录包含 `center_medium.png` 与26张目标视角图像。训练脚本中 `dataset_repeat=2`，预处理结果为1040条缓存记录；每个 epoch 为2080个优化 step。

| 参数 | 实际值 |
| --- | --- |
| LoRA rank | 64 |
| learning rate | 1e-4 |
| mixed precision | BF16 |
| gradient checkpointing | 开启 |
| max pixels | 1048576 |
| checkpoint interval | 5200 steps（约2.5 epoch） |
| 设备 | 单卡 A100 40GB |

LoRA 作用于 DiT 的 attention/FFN 相关线性层。训练期对 DiT 使用 `torch.compile(model, mode="reduce-overhead", fullgraph=False)`；`inject_compile.py` 为实际注入补丁。导出推理权重前，`clean_9b_lora_prefixes.py` 只移除 `_orig_mod.` 键名前缀，不改变张量值。

## 脚本与执行顺序

历史脚本使用服务器上的绝对路径（如 `/root/prep_dataset.py`、DiffSynth Studio 工作目录），迁移到另一环境时应按实际位置替换；参数和调用顺序保持不变。

1. `inject_compile.py`：对 DiffSynth Studio 的 `examples/flux2/model_training/train.py` 注入 `torch.compile` 调用；应在数据缓存和训练前执行一次。
2. `phase1_cache.sh`：调用 `prep_dataset.py`，以 `sft:data_process` 生成训练缓存。
3. `phase2_e35.sh`：执行 `sft:train`，从基座训练至35 epoch（72800 steps）。
4. `clean_9b_lora_prefixes.py`：将 E35 的 `step-72800.safetensors` 清理为可被未编译模型加载的 LoRA。
5. `phase2_e35_resume.sh`：加载清理后的E35 LoRA，继续相同数据和主要参数训练。该脚本的 `num_epochs=100000` 只是为了不设自然终点，实际人工在E65停止。

两阶段均为单卡训练。

## 阶段与非严格续训

前期队伍的线上得分与排名均较低，故原计划训练到 Epoch 35后服务器到期就及时止损。然而E35检查点表现超出预期，这促使队伍续租服务器，但训练已经中止，只得从 E35 LoRA 权重继续训练。

续训只加载清理后的 E35 LoRA 参数，未恢复 optimizer、learning-rate scheduler、dataloader 或随机数状态，因此不是严格断点恢复。

单卡A100条件下，训练稳定速度约2.92–2.94秒/step，E0–E65累计纯训练约110小时（均为基于稳定 step 速度的估算，而非完整墙钟计时）。

E40 在相同推理参数下明显低于中断前的 E35；E45 已恢复到接近 E35，E50 则超过 E35。这组曲线体现了仅加载 LoRA 参数续训后的早期状态下滑和后续恢复。E55、E60 又出现波动，E65 最终达到 1.8987。

据此估计，若在同一训练进程中连续训练，可以减少重新加载并适应训练状态的成本，总训练时间也应显著缩短；由于未进行连续训练对照实验，这一点只作为基于现有检查点曲线的判断，不作为实测结论。

以下为相同20 steps、CFG1、seed0下的线上结果：

| 检查点 | 总分 | PSNR | SSIM | LPIPS | NRIQA |
| --- | ---: | ---: | ---: | ---: | ---: |
| E35 | 1.8766 | 0.2871 | 0.3528 | 0.4545 | 0.7822 |
| E40 | 1.8527 | 0.2815 | 0.3444 | 0.4421 | 0.7847 |
| E45 | 1.8731 | 0.2877 | 0.3480 | 0.4599 | 0.7776 |
| E50 | 1.8936 | 0.2917 | 0.3532 | 0.4673 | 0.7813 |
| E55 | 1.8892 | 0.2907 | 0.3516 | 0.4662 | 0.7807 |
| E60 | 1.8800 | 0.2878 | 0.3459 | 0.4646 | 0.7817 |
| E65 | 1.8987 | 0.2933 | 0.3546 | 0.4717 | 0.7791 |

E40–E45的早期回落，与只恢复LoRA参数的非严格端点续训事实相符；上述观测不构成连续无中断训练的对照结论。


