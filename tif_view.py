import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import glob
import os

def visualize_pixel_json_on_image(image_path, pixel_json_path, show_labels=True, opacity=0.4):
    # Load image (assumes RGB or grayscale GeoTIFF)
    with rasterio.open(image_path) as src:
        # img = src.read()  # Shape: (bands, H, W)
        # img = np.moveaxis(img, 0, -1)  # to (H, W, bands)
        r,g,b = src.read(3),src.read(2),src.read(1)
        print(r.shape)
        img = np.stack([r,g,b],axis=-1)
        # img = img.reshape((img.shape[2],img.shape[1],img.shape[0]),axis=-1)
        print(img.shape)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)  # grayscale to RGB
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    overlay = img.copy()

    
    with open(pixel_json_path, 'r') as f:
        pixel_data = json.load(f)

    
    class_colors = {}
    color_palette = [
        (255, 0, 0),      
        (0, 255, 0),      
        (0, 0, 255),      
        (255, 255, 0),    
        (255, 0, 255),    
        (0, 255, 255),    
        (128, 128, 0),    
        (128, 0, 128),    
        (0, 128, 128),    
    ] 

    for i, feature in enumerate(pixel_data["features"]):
        class_name = feature.get("properties", {}).get("Class Name", f"Class-{i}")
        print(class_name)
        if class_name not in class_colors:
            class_colors[class_name] = color_palette[i % len(color_palette)]
        color = class_colors[class_name]

        for multipoly in feature["geometry"]["coordinates"]:
            for ring in multipoly:
                pts = np.array(ring, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
                cv2.fillPoly(overlay, [pts], color=color)

                # Put class label at the first point
                if show_labels and len(pts) > 0:
                    text_pos = tuple(pts[0][0])
                    cv2.putText(overlay, class_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Blend original and overlay
    blended = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)

    # Show with matplotlib
    plt.figure(figsize=(36, 30))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))  
    plt.title("Pixel Annotation Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def visualize_tiffs_with_json(tiff_dir, show_labels, opacity):
    tiffs = glob.glob(tiff_dir+'/*.tif')
    for tiff in tiffs:
        print(tiff)
        json_file = os.path.splitext(tiff)[0]+'.json'
        visualize_pixel_json_on_image(
            image_path=tiff,
            pixel_json_path=json_file,
            show_labels=show_labels,
            opacity=opacity
        )


def main():
    visualize_tiffs_with_json(tiff_dir="PS03_1/Datasets/sample-set/MetroShed,STP & Sheds", show_labels=True, opacity=0.5)

if __name__ == "__main__":
    main()