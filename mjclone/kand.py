from kandinsky2 import get_kandinsky2_1
from PIL import Image


def fuse_images(img1_path: str, img2_path: str):
    model = get_kandinsky2_1("cuda", task_type="text2img")
    images_texts = ["shrek", Image.open(img1_path), "gigachad", Image.open(img2_path)]
    weights = [
        0.25,
        0.25,
        0.25,
        0.25,
    ]
    images = model.mix_images(
        images_texts,
        weights,
        num_steps=50,
        batch_size=1,
        guidance_scale=10,
        h=500,
        w=500,
        sampler="p_sampler",
        prior_cf_scale=4,
        prior_steps="5",
    )
    return images
