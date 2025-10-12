import os
import csv
import random
from itertools import combinations, product

def generate_pairs_csv(root_dir, class_subdirs, output_csv, num_dissimilar_per_class=90, seed=42):
    random.seed(seed)
    all_classes = {}
    for cls in class_subdirs:
        class_path = os.path.join(root_dir, cls)
        images = [os.path.join(class_path, f) for f in os.listdir(class_path)
                  if f.lower().endswith('.png')]
        all_classes[cls] = images

    pairs = []

    for cls, images in all_classes.items():
        for img1, img2 in combinations(images, 2):
            pairs.append((img1, img2, 1))

    class_names = list(all_classes.keys())
    for cls in class_names:
        other_classes = [c for c in class_names if c != cls]
        img_samples = all_classes[cls]
        for _ in range(num_dissimilar_per_class):
            img1 = random.choice(img_samples)
            other_cls = random.choice(other_classes)
            img2 = random.choice(all_classes[other_cls])
            pairs.append((img1, img2, 0))

    random.shuffle(pairs)
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['img1', 'img2', 'label'])
        writer.writerows(pairs)

    print(f"CSV file saved: {output_csv}")
    print(f"Total pairs: {len(pairs)}")
    similar = sum(1 for p in pairs if p[2] == 1)
    dissimilar = len(pairs) - similar
    print(f"Similar pairs: {similar}, Dissimilar pairs: {dissimilar}")


if __name__ == "__main__":
    root_dir = "/home/zerodegree/Desktop/PS-3/PS03_1/Datasets/Classes"  
    class_subdirs = ["Brick Kiln","Metro Shed","Play Ground","Pond-1","Pond-2","Sheds","Solar Panel","STP"]  
    output_csv = "pair_input.csv"

    generate_pairs_csv(root_dir, class_subdirs, output_csv)
