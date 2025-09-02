import json
import numpy as np
from PIL import Image, ImageDraw

# input files
image_path = "PS03_1/Datasets/sample-set/Brick Kiln/GC01PS03D0011.jpg"
json_path = "PS03_1/Datasets/sample-set/Brick Kiln/GC01PS03D0011.json"

# load image
img = Image.open(image_path)

# load JSON
with open(json_path, "r") as f:
    data = json.load(f)

for i, feature in enumerate(data["features"]):
    coords = feature["geometry"]["coordinates"][0][0]  # first polygon ring
    polygon = [(x, y) for x, y in coords]              # list of (x,y) tuples

    # create mask for polygon
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask)

    # apply mask
    out = np.array(img) * mask[:, :, None]

    # crop bounding box
    ys, xs = np.where(mask)
    if ys.size and xs.size:
        cropped = out[min(ys):max(ys), min(xs):max(xs)]
        cropped_img = Image.fromarray(cropped)
        cropped_img.save(f"crop_{i+1}.png")
