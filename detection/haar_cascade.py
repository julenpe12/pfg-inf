#==================== Results ====================
#Easy   Val AP: 0.36172934085919306
#Medium Val AP: 0.32319682016352574
#Hard   Val AP: 0.1549213808443529
#=================================================

import os
import cv2

def main():
    # 1) Path to the WIDER Face validation or test images
    wider_val_images = r"dataset\WIDER_val\images"

    # 2) Output folder for .txt detection files
    output_results_dir = r"results\haar_cascade"

    if not os.path.exists(output_results_dir):
        os.makedirs(output_results_dir)

    # 3) Initialize Haar Cascade
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # 4) Gather subfolders: e.g., 0--Parade, 1--Handshaking, etc.
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

        # Get image files
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

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ----------------------------------------------------------------------------------
            # detectMultiScale2 -> returns bounding boxes AND levelWeights (pseudo confidences).
            # If your OpenCV doesn't have detectMultiScale2, see the note at the end.
            # ----------------------------------------------------------------------------------
            bboxes, weights = face_cascade.detectMultiScale2(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Prepare the result file path
            basename, _ = os.path.splitext(image_name)
            result_file_path = os.path.join(subfolder_out, basename + ".txt")

            # WIDER FACE submission format:
            #
            # Line 1: "subfolder/image_name"  (the full path to the image)
            # Line 2: <number_of_faces>
            # Lines 3+: left top width height score
            #
            # Example:
            #   0--Parade/0_Parade_marchingband_1_5.jpg
            #   2
            #   23 40 70 80 0.95
            #   120 45 65 75 0.88

            with open(result_file_path, 'w') as f:
                # 1) image path
                f.write(f"{subfolder}/{image_name}\n")

                # 2) number of faces
                f.write(f"{len(bboxes)}\n")

                # 3) bounding boxes
                for (box, score) in zip(bboxes, weights):
                    x, y, w, h = box
                    # 'score' is the pseudo-confidence from detectMultiScale2
                    f.write(f"{x} {y} {w} {h} {float(score)}\n")

    print("Detection results (submission format) have been written.")
    print("Ready to submit to the WIDER Face server or run local evaluation on the test set.")


if __name__ == "__main__":
    main()