from mjclone.models.txt2img.base import Txt2ImgModule
from kandinsky2 import get_kandinsky2

from mjclone.models.utils import flush


class KandinskyTxt2Img(Txt2ImgModule):
    def __init__(self) -> None:
        super().__init__()

    def create_img(self, images: list, prompts: list):
        flush()
        self.model = get_kandinsky2(
            "cuda", task_type="text2img", model_version="2.1", use_flash_attention=False
        )
        images_texts = images + prompts
        images.extend(prompts)
        weights = [
            0.3,
            0.3,
            0.2,
            0.2,
        ]
        return self.model.mix_images(
            images_texts,
            weights,
            num_steps=100,
            batch_size=1,
            guidance_scale=6,
            h=512,
            w=512,
            sampler="ddim_sampler",
            prior_cf_scale=4,
            prior_steps="5",
            negative_prior_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        )
