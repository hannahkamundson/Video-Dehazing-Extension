import numpy as np
import os
from PIL import Image


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

# get the path/directory


def pix_reduce(image_dir, reduce_dir):
    for _, dirr in enumerate(listdir_nohidden(image_dir)):
        in_dir = os.path.join(image_dir, dirr)
        os.makedirs(os.path.join(reduce_dir, dirr))
        for i, image in enumerate(sorted(listdir_nohidden(in_dir))):
            # check if the image ends with png
            if (image.endswith(".JPG")):
                img = Image.open(os.path.join(in_dir, image))
                img_resized = img.resize((256, 256))
                if i % 10 == 0:
                    print(i, f"New size : {img_resized.size}")
                img_resized.save(os.path.join(reduce_dir, dirr, image))


if __name__ == "__main__":
    inn = '/scratch/08310/rs821505/Video-Dehazing-Extension/dataset/REVIDE/Train/gt'
    out = '/scratch/08310/rs821505/Video-Dehazing-Extension/dataset/REVIDE_REDUCED/Train/gt'
    pix_reduce(inn,out)
