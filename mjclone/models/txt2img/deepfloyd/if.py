from dataclasses import dataclass
from typing import Self

import torch
from diffusers import (
    DiffusionPipeline,
    IFImg2ImgPipeline,
    IFImg2ImgSuperResolutionPipeline,
)
from diffusers.utils import pt_to_pil

from mjclone.utils import flush


@dataclass(slots=True)
class IfConfig:
    stage_1_weights: str = "DeepFloyd/IF-I-L-v1.0"
    stage_2_weights: str = "DeepFloyd/IF-II-L-v1.0"
    stage_3_weights: str = "stabilityai/stable-diffusion-x4-upscaler"


class IfModule:
    def __init__(self, stage_1, stage_2, stage_3) -> None:
        self.stage_1 = stage_1
        self.stage_2 = stage_2
        self.stage_3 = stage_3

        self.stage_1.enable_model_cpu_offload()
        self.stage_2.enable_model_cpu_offload()
        self.stage_3.enable_model_cpu_offload()

        self.generator = torch.manual_seed(1)

    @classmethod
    def from_config(cls, config: IfConfig) -> Self:
        # stage_1
        stage_1 = IFImg2ImgPipeline.from_pretrained(
            config.stage_1_weights,
            load_in_8bit=True,
            variant="8bit",
            torch_dtype=torch.float16,
        )

        # stage 2
        stage_2 = IFImg2ImgSuperResolutionPipeline.from_pretrained(
            config.stage_2_weights,
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
        )

        # stage 3
        safety_modules = {
            "feature_extractor": stage_1.feature_extractor,
            "safety_checker": stage_1.safety_checker,
            "watermarker": stage_1.watermarker,
        }

        stage_3 = DiffusionPipeline.from_pretrained(
            config.stage_3_weights,
            **safety_modules,
            torch_dtype=torch.float16,
        )
        return cls(stage_1, stage_2, stage_3)

    def generate(self, original_image, prompt: str):
        image, prompt_embeds, negative_embeds = self.run_stage_1(original_image, prompt)
        image = self.run_stage_2(original_image, image, prompt_embeds, negative_embeds)
        self.run_stage_3(image, prompt)

    def run_stage_1(self, original_image, prompt):
        # stage 1
        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(prompt)
        image = self.stage_1(
            image=original_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=self.generator,
            output_type="pt",
        ).images
        pt_to_pil(image)[0].save("./if_stage_I.png")
        flush()
        return image, prompt_embeds, negative_embeds

    def run_stage_2(self, original_image, image, prompt_embeds, negative_embeds):
        image = self.stage_2(
            image=image,
            original_image=original_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=self.generator,
            output_type="pt",
        ).images
        pt_to_pil(image)[0].save("./if_stage_II.png")
        flush()
        return image

    def run_stage_3(self, image, prompt):
        image = self.stage_3(
            prompt=prompt, image=image, generator=self.generator, noise_level=100
        ).images
        image[0].save("./if_stage_III.png")
        flush()
