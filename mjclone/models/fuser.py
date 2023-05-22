from dataclasses import dataclass

from huggingface_hub import login
from mjclone.models.img2text.interrogator.model import ClipInterrogator
from mjclone.models.txt2img.kand.model import KandinskyTxt2Img
from mjclone.models.utils import flush


class ImageFuser:
    def __init__(
        self,
        img2txt=ClipInterrogator.from_config(),
        txt2img=KandinskyTxt2Img(),
    ) -> None:
        flush()
        self.img2txt = img2txt
        self.txt2img = txt2img

    def run_img2txt(self, images) -> list[str]:
        out = []
        for image in images:
            out.append(self.img2txt.generate_txt(image))
        return out

    def run_txt2img(self, images: list, prompts: list[str]):
        image = self.txt2img.create_img(images, prompts)
        return image

    def fuse(self, original_image, target_image):
        prompt = self.run_img2txt([original_image, target_image])
        return self.run_txt2img([original_image, target_image], prompt)
