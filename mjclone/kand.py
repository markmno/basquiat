from kandinsky2.kandinsky2_1_model import Kandinsky2_1
from PIL import Image


def fuse_images(img1_path: str, img2_path: str):
    kand = Kandinsky2_1()

    model = get_kandinsky2(
        "cpu", task_type="text2img", model_version="2.1", use_flash_attention=True
    )
    images_texts = [Image.open("img1.jpg"), Image.open("img2.jpg")]
    weights = [
        0.5,
        0.5,
    ]
    images = model.mix_images(
        images_texts,
        weights,
        num_steps=150,
        batch_size=1,
        guidance_scale=5,
        h=768,
        w=768,
        sampler="p_sampler",
        prior_cf_scale=4,
        prior_steps="5",
    )
    return images
