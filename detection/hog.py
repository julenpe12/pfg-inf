#==================== Results ====================
#Easy   Val AP: 0.49409882304308883
#Medium Val AP: 0.39863730650658075
#Hard   Val AP: 0.1674631413741721
#=================================================

import os
import cv2
import dlib

def main():
    # 1) Path to the WIDER Face images (val or test)
    wider_val_images = r"dataset\WIDER_val\images"

    # 2) Output folder for .txt detection files
    output_results_dir = r"results\dlib_hog"

    if not os.path.exists(output_results_dir):
        os.makedirs(output_results_dir)

    # 3) Initialize Dlib HOG face detector
    hog = dlib.get_frontal_face_detector()

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

            # ------------------------------------------------------------------------
            # Use hog.run(...) to get bounding boxes (dets) & confidence (scores).
            # upsample_num_times = 1 or 2 for better detection on small faces (but slower).
            # adjust_threshold <= 0.0 means we include lower-confidence detections too.
            # ------------------------------------------------------------------------
            dets, scores, _ = hog.run(gray, upsample_num_times=1)

            # Prepare the result file path
            basename, _ = os.path.splitext(image_name)
            result_file_path = os.path.join(subfolder_out, basename + ".txt")

            # WIDER FACE submission format:
            #
            #   1) line 1: "subfolder/image_name"
            #   2) line 2: number_of_faces
            #   3) lines 3+ : left top width height score
            #
            # e.g.
            #  0--Parade/0_Parade_marchingband_1_5.jpg
            #  2
            #  23 40 70 80 0.95
            #  120 45 65 75 0.88

            with open(result_file_path, 'w') as f:
                # (1) image path
                f.write(f"{subfolder}/{image_name}\n")

                # (2) number of faces
                f.write(f"{len(dets)}\n")

                # (3) bounding boxes
                for rect, sc in zip(dets, scores):
                    left = rect.left()
                    top = rect.top()
                    width = rect.right() - left
                    height = rect.bottom() - top
                    score = float(sc)  # convert from dlib's double to float
                    f.write(f"{left} {top} {width} {height} {score}\n")

    print("Dlib HOG detection results (submission format) have been saved.")
    print("You can now run the WIDER Face evaluation or submit them as required.")

if __name__ == "__main__":
    main()