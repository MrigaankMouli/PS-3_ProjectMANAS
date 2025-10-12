import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import glob
import os

def visualize_detections_on_image(image_path, ground_truth_json_path, detections_json_path=None, 
                                show_labels=True, gt_opacity=0.3, det_opacity=0.5):

    with rasterio.open(image_path) as src:
        r, g, b = src.read(3), src.read(2), src.read(1)
        print(f"Image shape: {r.shape}")
        img = np.stack([r, g, b], axis=-1)
        print(f"Stacked image shape: {img.shape}")
        
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)  
        
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    gt_overlay = img.copy()
    det_overlay = img.copy()
    
    gt_colors = {
        'Brick Kiln': (0, 255, 0),  
        'default': (0, 255, 0)
    }
    
    det_colors = {
        'high_conf': (255, 0, 0),    
        'medium_conf': (255, 165, 0), 
        'low_conf': (255, 255, 0),   
        'false_pos': (255, 0, 255)   
    }
    
    if os.path.exists(ground_truth_json_path):
        with open(ground_truth_json_path, 'r') as f:
            gt_data = json.load(f)
        
        print(f"Drawing {len(gt_data['features'])} ground truth annotations...")
        
        for i, feature in enumerate(gt_data["features"]):
            class_name = feature.get("properties", {}).get("Class Name", f"Class-{i}")
            print(f"GT Class: {class_name}")
            
            color = gt_colors.get(class_name, gt_colors['default'])
            
            for multipoly in feature["geometry"]["coordinates"]:
                for ring in multipoly:
                    pts = np.array(ring, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    
                    cv2.fillPoly(gt_overlay, [pts], color=color)
                    cv2.polylines(gt_overlay, [pts], isClosed=True, color=color, thickness=3)
                    
                    if show_labels and len(pts) > 0:
                        text_pos = tuple(pts[0][0])
                        cv2.putText(gt_overlay, f"GT: {class_name}", text_pos, 
                                  cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, 
                                  color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    
    if detections_json_path and os.path.exists(detections_json_path):
        with open(detections_json_path, 'r') as f:
            det_data = json.load(f)
        
        detections = det_data.get('detections', [])
        print(f"Drawing {len(detections)} detections...")
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']  
            confidence = detection['confidence']
            class_name = detection.get('class', 'Unknown')
            
            x_min, y_min, x_max, y_max = map(int, bbox)
            
            if confidence > 0.7:
                color = det_colors['high_conf']
                conf_label = "HIGH"
            elif confidence > 0.4:
                color = det_colors['medium_conf'] 
                conf_label = "MED"
            else:
                color = det_colors['low_conf']
                conf_label = "LOW"
            
            thickness = max(2, int(confidence * 4))
            cv2.rectangle(det_overlay, (x_min, y_min), (x_max, y_max), color, thickness)
            
            if show_labels:
                label = f"PRED: {class_name} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                cv2.rectangle(det_overlay, (x_min, y_min - label_size[1] - 10), 
                            (x_min + label_size[0] + 10, y_min), color, -1)
                
                cv2.putText(det_overlay, label, (x_min + 5, y_min - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                          color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    

    blended = cv2.addWeighted(gt_overlay, gt_opacity, img, 1 - gt_opacity, 0)
    
    if detections_json_path and os.path.exists(detections_json_path):
        mask = np.zeros_like(det_overlay, dtype=np.uint8)
        cv2.subtract(det_overlay, img, mask)
        det_only = cv2.add(img, mask)
        blended = cv2.addWeighted(blended, 1, det_only, det_opacity, 0)
    
    legend_height = 150
    legend_width = 400
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    
    cv2.rectangle(legend, (10, 10), (30, 30), gt_colors['Brick Kiln'], -1)
    cv2.putText(legend, "Ground Truth", (40, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset = 40
    for conf_type, color in det_colors.items():
        if conf_type == 'false_pos':
            continue
        cv2.rectangle(legend, (10, y_offset), (30, y_offset + 20), color, 2)
        y_offset += 30
    
    legend_y = blended.shape[0] - legend_height - 10
    legend_x = 10
    blended[legend_y:legend_y + legend_height, legend_x:legend_x + legend_width] = legend
    
    plt.figure(figsize=(20, 16))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    
    title_parts = ["Ground Truth (Green) vs Detections (Colored Boxes)"]
    if detections_json_path and os.path.exists(detections_json_path):
        with open(detections_json_path, 'r') as f:
            det_data = json.load(f)
        num_dets = len(det_data.get('detections', []))
        title_parts.append(f" | {num_dets} Detections")
    
    plt.title(" ".join(title_parts), fontsize=14, pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def visualize_tiffs_with_detections(tiff_dir, detections_dir=None, show_labels=True, 
                                  gt_opacity=0.3, det_opacity=0.5):
    tiffs = glob.glob(tiff_dir + '/*.tif')
    
    for tiff in tiffs:
        print(f"\nProcessing: {tiff}")
        
        gt_json_file = os.path.splitext(tiff)[0] + '.json'
        
        base_name = os.path.splitext(os.path.basename(tiff))[0]
        if detections_dir:
            det_json_file = os.path.join(detections_dir, f"{base_name}_detections.json")
        
        if not os.path.exists(gt_json_file):
            print(f"Warning: Ground truth file not found: {gt_json_file}")
            continue
            
        if not os.path.exists(det_json_file):
            print(f"Warning: Detection file not found: {det_json_file}")
            det_json_file = None
        
        visualize_detections_on_image(
            image_path=tiff,
            ground_truth_json_path=gt_json_file,
            detections_json_path=det_json_file,
            show_labels=show_labels,
            gt_opacity=gt_opacity,
            det_opacity=det_opacity
        )

def main():
    """
    Main function - modify these paths as needed
    """

    visualize_tiffs_with_detections(
        tiff_dir="PS03_1/Datasets/sample-set/Brick Kiln",
        detections_dir='PS03_1/Datasets/sample-set/Brick Kiln',  
        show_labels=True,
        gt_opacity=0.8,  
        det_opacity=0.6  
    )

if __name__ == "__main__":
    main()