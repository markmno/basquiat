from typing import Self
import torch
from clip_interrogator import Config, Interrogator

from mjclone.models.utils import flush

from ..base import Img2TxtModule


class ClipInterrogator(Img2TxtModule):
    def __init__(self, interrogator: Interrogator) -> None:
        self.interrogator = interrogator

    @classmethod
    def from_config(
        cls,
        config=Config(
            flavor_intermediate_count=16,
            clip_model_name="ViT-L-14/openai",
            caption_model_name="blip-large",
            device="cuda",
            chunk_size=512,
        ),
    ) -> Self:
        interrogator = Interrogator(config)
        return cls(interrogator)

    def generate_txt(self, image) -> str:
        flush()
        text = self.interrogator.interrogate(image)
        flush()
        return text
