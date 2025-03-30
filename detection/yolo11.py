#==================== Results ====================
#Easy   Val AP: 0.9571999894809975
#Medium Val AP: 0.9368587990113666
#Hard   Val AP: 0.7873753831886807
#=================================================

import os
import cv2
from ultralytics import YOLO

def main():
    # 1) Path to WIDER Face validation images
    wider_val_images = r"dataset\WIDER_val\images"  # <--- Adjust to your path

    # 2) Output folder for .txt detection files
    output_results_dir = r"results\yolov5s"
    if not os.path.exists(output_results_dir):
        os.makedirs(output_results_dir)

    # 3) Load YOLOv11-l model
    model_path = "models/face_detection_yolov5su.pt"
    model = YOLO(model_path)
    model.to("cuda")

    # 4) Gather subfolders (like 0--Parade, 1--Handshaking, etc.)
    subfolders = [
        d for d in os.listdir(wider_val_images)
        if os.path.isdir(os.path.join(wider_val_images, d))
    ]

    for subfolder in subfolders:
        subfolder_path = os.path.join(wider_val_images, subfolder)

        # Make sure output subfolder exists
        subfolder_out = os.path.join(output_results_dir, subfolder)
        if not os.path.exists(subfolder_out):
            os.makedirs(subfolder_out)

        image_files = [
            f for f in os.listdir(subfolder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        image_files.sort()

        print(f"Processing folder: {subfolder} with {len(image_files)} images")

        for image_name in image_files:
            image_path = os.path.join(subfolder_path, image_name)
            img = cv2.imread(image_path)
            if img is None:
                continue

            # 5) Detect with YOLO
            results = model(img)  # get predictions
            preds = results[0]    # first batch prediction

            # 6) Prepare submission format
            basename, _ = os.path.splitext(image_name)
            result_file_path = os.path.join(subfolder_out, basename + ".txt")

            with open(result_file_path, 'w') as f:
                # line 1: "subfolder/image_name"
                f.write(f"{subfolder}/{image_name}\n")

                # line 2: number of faces (or total detections)
                num_boxes = len(preds.boxes)
                f.write(f"{num_boxes}\n")

                # lines 3+: each bounding box as [left top width height score]
                for box in preds.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # xyxy
                    score = float(box.conf[0])
                    width = x2 - x1
                    height = y2 - y1

                    f.write(f"{int(x1)} {int(y1)} {int(width)} {int(height)} {score}\n")

    print("YOLOv11-l detection results for WIDER Face have been saved (submission format).")
    print("You can run the WIDER Face evaluation or submit these files as needed.")

if __name__ == "__main__":
    main()