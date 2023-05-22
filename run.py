from PIL import Image
import argparse
from mjclone.models.fuser import ImageFuser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image comparison script")
    parser.add_argument("image1_path", type=str, help="Path to the first image")
    parser.add_argument("image2_path", type=str, help="Path to the second image")

    args = parser.parse_args()

    img1_path = args.image1_path
    img2_path = args.image2_path

    source_image = Image.open(img1_path).convert("RGB")
    target_image = Image.open(img2_path).convert("RGB")
    fuser = ImageFuser()
    ret = fuser.fuse(source_image, target_image)
    ret[0].save(f"assets/img_{2}.png")
