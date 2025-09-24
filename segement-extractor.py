import os
import json
import cv2
import numpy as np

input_dir = "PS03_1/Datasets/sample-set/Brick Kiln"      
output_dir = "PS03_1/Datasets/sample-set/Brick Kiln/Segments"  
os.makedirs(output_dir, exist_ok=True)

files = os.listdir(input_dir)

for file in files:
    if file.endswith(".png") or file.endswith(".jpg"):
        img_path = os.path.join(input_dir, file)
        base_name = os.path.splitext(file)[0]
        json_path = os.path.join(input_dir, f"{base_name}.json")
        
        image = cv2.imread(img_path)
        with open(json_path) as f:
            data = json.load(f)
        
        for idx, feature in enumerate(data['features']):
            polygons = feature['geometry']['coordinates']
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            for poly_group in polygons:
                for poly in poly_group:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
            
            segmented = cv2.bitwise_and(image, image, mask=mask)
            
            ys, xs = np.where(mask>0)
            if len(xs) == 0 or len(ys) == 0:
                continue  
            x, y, w, h = xs.min(), ys.min(), xs.max()-xs.min(), ys.max()-ys.min()
            cropped_segment = segmented[y:y+h, x:x+w]
            
            out_name = f"{base_name}_segment_{idx+1}.png"
            cv2.imwrite(os.path.join(output_dir, out_name), cropped_segment)

print("All images processed!")
