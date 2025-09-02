from ultralytics import YOLO
import cv2
import torch
import numpy as np
from typing import List, Dict, Tuple

class SatelliteObjectDetector:
    def __init__(self, model_path: str = None):
        if model_path:
            self.model = YOLO(model_path)  
        else:
            self.model = YOLO('yolov8n.pt')  
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def train_on_satellite_data(self, data_yaml_path: str, epochs: int = 100):
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            device=self.device,
            batch=16,
            name='satellite_sheds',
            hsv_h=0.1,      
            hsv_s=0.3,      
            hsv_v=0.3,      
            degrees=45,     
            translate=0.1,  
            scale=0.2,      
            mosaic=1.0,     
            mixup=0.1       
        )
        return results
    
    def detect_objects(self, image_path: str, conf_threshold: float = 0.5) -> List[Dict]:
        results = self.model(image_path, conf=conf_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'area': (x2-x1) * (y2-y1)
                    })
        
        return detections
    
    def extract_object_crops(self, image_path: str, detections: List[Dict]) -> List[np.ndarray]:
        """Extract cropped regions of detected objects"""
        image = cv2.imread(image_path)
        crops = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        
        return crops

def prepare_yolo_dataset(annotations_json: str, images_dir: str, output_dir: str):
    """Convert your annotations to YOLO format"""
    import json
    import os
    from pathlib import Path
    
    for split in ['train', 'val']:
        Path(f"{output_dir}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/labels/{split}").mkdir(parents=True, exist_ok=True)
    
    with open(annotations_json, 'r') as f:
        annotations = json.load(f)
    
    class_names = ['Sheds']  
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    for idx, annotation in enumerate(annotations):
        image_name = annotation['image_name']
        image_path = f"{images_dir}/{image_name}"
        
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        img_height, img_width = img.shape[:2]
        
        yolo_annotations = []
        
        for obj in annotation.get('objects', []):
            class_name = obj['class_name']
            if class_name not in class_to_id:
                continue
                
            class_id = class_to_id[class_name]
            
            if 'bbox' in obj:
                x, y, w, h = obj['bbox']
            else:
                x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
                x, y, w, h = x1, y1, x2-x1, y2-y1
            
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width_norm = w / img_width
            height_norm = h / img_height
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
        
        split = 'train' if idx % 5 != 0 else 'val'
        
        import shutil
        shutil.copy(image_path, f"{output_dir}/images/{split}/{image_name}")
        
        label_path = f"{output_dir}/labels/{split}/{Path(image_name).stem}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
    data_yaml = f"""
path: {output_dir}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
    
    with open(f"{output_dir}/data.yaml", 'w') as f:
        f.write(data_yaml)
    
    print(f"Dataset prepared in {output_dir}")
    print(f"Classes: {class_names}")