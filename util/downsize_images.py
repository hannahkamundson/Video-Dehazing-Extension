import sys
from PIL import Image
import os
import glob


# read in from/to paths
assert len(sys.argv) == 3, "You need a from command and a to command"
from_folder = sys.argv[1]
to_folder = sys.argv[2]

newsize = (64, 64)

for file in glob.glob(os.path.join(from_folder, "**", "*.JPG"), recursive = True):
    im = Image.open(os.path.abspath(file))
    im1 = im1.resize(newsize)
    new_path = os.join(to_folder, os.path.basename(file))
    im1.save(new_path)
