"""Prompt-folded 9B L75 LoRA: 20 steps, CFG 1.0, single seed 42."""

import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open


MODEL_BASE_PATH = "/workspace/multiAngleSynthesis/diffSynth_studio/models"
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
LORA_PATH = "./lora/e50_e65_l75.safetensors"
PROMPT_CONSTANTS_PATH = "prompt_constants/view_prompt_constants.safetensors"
NUM_INFERENCE_STEPS = 20
CFG_SCALE = 1.0
SEED = 42

VIEW_FILES = [
    "center_far.png", "center_near.png",
    "top_far.png", "top_medium.png", "top_near.png",
    "top_left_far.png", "top_left_medium.png", "top_left_near.png",
    "top_right_far.png", "top_right_medium.png", "top_right_near.png",
    "bottom_far.png", "bottom_medium.png", "bottom_near.png",
    "bottom_left_far.png", "bottom_left_medium.png", "bottom_left_near.png",
    "bottom_right_far.png", "bottom_right_medium.png", "bottom_right_near.png",
    "left_far.png", "left_medium.png", "left_near.png",
    "right_far.png", "right_medium.png", "right_near.png",
]
assert len(VIEW_FILES) == 26


def filename_to_prompt(filename: str) -> str:
    return " ".join(Path(filename).stem.split("_"))


class PromptConstants:
    """Frozen BF16 conditioning for the fixed 26 view prompts."""

    def __init__(self, path):
        self.path = Path(path)
        self._conditioning = {}
        with safe_open(str(self.path), framework="pt", device="cpu") as archive:
            metadata = archive.metadata()
            if metadata.get("model_id") != MODEL_ID:
                raise ValueError(f"Prompt constants model mismatch: {metadata.get('model_id')!r}")
            prompts = json.loads(metadata["prompts"])
            for prompt in prompts:
                self._conditioning[prompt] = (
                    archive.get_tensor(f"embedding::{prompt}"),
                    archive.get_tensor(f"text_ids::{prompt}"),
                )

    @property
    def prompts(self):
        return set(self._conditioning)

    def get(self, prompt):
        try:
            return self._conditioning[prompt]
        except KeyError as error:
            raise KeyError(f"No frozen conditioning for prompt: {prompt!r}") from error


def install_prompt_folded_units(pipe, flux2_module, constants):
    """Replace runtime text encoding with frozen single-sample conditioning."""

    class NoopPromptEmbedder(flux2_module.Flux2Unit_PromptEmbedder):
        def process(self, pipe, prompt):
            return {}

    class FrozenQwen3PromptEmbedder(flux2_module.Flux2Unit_Qwen3PromptEmbedder):
        def process(self, pipe, prompt):
            prompt_embeds, text_ids = constants.get(prompt)
            return {
                "prompt_embeds": prompt_embeds.to(device=pipe.device, dtype=pipe.torch_dtype),
                "text_ids": text_ids.to(device=pipe.device),
            }

    replacements = {
        flux2_module.Flux2Unit_PromptEmbedder: NoopPromptEmbedder,
        flux2_module.Flux2Unit_Qwen3PromptEmbedder: FrozenQwen3PromptEmbedder,
    }
    pipe.units = [replacements[type(unit)]() if type(unit) in replacements else unit for unit in pipe.units]


class ImageGenerator:
    def __init__(self):
        self.pipe = None
        self.prompt_constants = None

    def load_model(self):
        if self.pipe is not None:
            return self.pipe

        from diffsynth.pipelines import flux2_image

        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = MODEL_BASE_PATH
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
        self.pipe = flux2_image.Flux2ImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                flux2_image.ModelConfig(model_id=MODEL_ID, origin_file_pattern="transformer/*.safetensors"),
                flux2_image.ModelConfig(model_id=MODEL_ID, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            tokenizer_config=None,
        )
        if not os.path.isfile(LORA_PATH):
            raise FileNotFoundError(f"Required LoRA weight is missing: {LORA_PATH}")
        self.pipe.load_lora(self.pipe.dit, flux2_image.ModelConfig(path=LORA_PATH))
        constants_path = Path(__file__).resolve().parent / PROMPT_CONSTANTS_PATH
        self.prompt_constants = PromptConstants(constants_path)
        install_prompt_folded_units(self.pipe, flux2_image, self.prompt_constants)
        return self.pipe

    def generate(self, center_image_path: str, output_dir: str):
        pipe = self.load_model()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        edit_image = Image.open(center_image_path).convert("RGB")

        started_at = time.time()
        with torch.inference_mode():
            for index, filename in enumerate(VIEW_FILES, start=1):
                image = pipe(
                    prompt=filename_to_prompt(filename),
                    edit_image=[edit_image],
                    seed=SEED,
                    rand_device="cuda",
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    cfg_scale=CFG_SCALE,
                    denoising_strength=1.0,
                    height=edit_image.height,
                    width=edit_image.width,
                )
                image.save(output_path / filename)
                print(f"[9B-L75-PROMPTFOLD-S20-SEED42] [{index}/26] {filename}", flush=True)
        elapsed = time.time() - started_at
        print(
            f"[9B-L75-PROMPTFOLD-S20-SEED42] done: {elapsed:.1f}s total; "
            f"{elapsed / 26:.2f}s/view",
            flush=True,
        )
