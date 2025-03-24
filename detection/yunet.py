#==================== Results ====================
#Easy   Val AP: 0.8759483172551658
#Medium Val AP: 0.8501302625333667
#Hard   Val AP: 0.7046548351053237
#=================================================
import os
import cv2

def main():
    # 1) Path to the WIDER Face validation images
    wider_val_images = r"dataset\WIDER_val\images"

    # 2) Output folder for .txt detection files
    output_results_dir = r"results\yunet"

    # Create the output directory if not present
    if not os.path.exists(output_results_dir):
        os.makedirs(output_results_dir)

    # 3) Load YuNet
    model_path = "models/face_detection_yunet_2023mar_int8.onnx"
    face_detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (640, 640),
        0.5,  # scoreThreshold
        0.5,  # nmsThreshold
        5000
    )
    

    # 4) Gather subfolders (e.g. 0--Parade, 1--Handshaking, etc.)
    subfolders = [
        d for d in os.listdir(wider_val_images)
        if os.path.isdir(os.path.join(wider_val_images, d))
    ]

    for subfolder in subfolders:
        subfolder_path = os.path.join(wider_val_images, subfolder)

        # Ensure the output subfolder exists
        subfolder_out = os.path.join(output_results_dir, subfolder)
        if not os.path.exists(subfolder_out):
            os.makedirs(subfolder_out)

        # List image files
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

            h, w = img.shape[:2]
            # Update input size for YuNet
            face_detector.setInputSize((w, h))

            # Detect faces
            rc, faces = face_detector.detect(img)

            # Prepare output text file path
            basename, _ = os.path.splitext(image_name)
            result_file_path = os.path.join(subfolder_out, basename + ".txt")

            with open(result_file_path, 'w') as f:
                # 1) The first line is "<subfolder>/<image_name>"
                f.write(f"{subfolder}/{image_name}\n")

                num_faces = 0
                if rc > 0 and faces is not None:
                    num_faces = faces.shape[0]

                # 2) The second line is the number of faces
                f.write(f"{num_faces}\n")

                # 3) Each subsequent line => [left top width height score]
                if num_faces > 0:
                    for face in faces:
                        # face[0:4] => [x, y, w, h]
                        # face[-1]  => confidence
                        x, y, box_w, box_h = face[:4].astype(int)
                        score = float(face[-1])
                        f.write(f"{x} {y} {box_w} {box_h} {score}\n")

    print("YuNet detection results for WIDER Face have been saved (submission format).")
    print("You can run the official WIDER evaluation code or submit these files as required.")


if __name__ == "__main__":
    main()