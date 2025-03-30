import os
import cv2
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

class FaceRecognizerLBPH:
    def __init__(self, recognition_threshold=50.0, grid_x=8, grid_y=8):
        self.recognition_threshold = recognition_threshold
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.grid_x = grid_x
        self.grid_y = grid_y

    def detect_and_crop_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        return face

    def compute_lbp(self, gray):
        # Compute LBP using a simple 3x3 neighborhood.
        h, w = gray.shape
        lbp = np.zeros((h-2, w-2), dtype=np.int32)
        center = gray[1:-1, 1:-1]
        lbp |= (gray[0:-2, 0:-2] >= center) << 7
        lbp |= (gray[0:-2, 1:-1] >= center) << 6
        lbp |= (gray[0:-2, 2:] >= center) << 5
        lbp |= (gray[1:-1, 2:] >= center) << 4
        lbp |= (gray[2:, 2:] >= center) << 3
        lbp |= (gray[2:, 1:-1] >= center) << 2
        lbp |= (gray[2:, 0:-2] >= center) << 1
        lbp |= (gray[1:-1, 0:-2] >= center) << 0
        return lbp

    def extract_features(self, image):
        face = self.detect_and_crop_face(image)
        if face is None:
            return None
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        lbp_image = self.compute_lbp(gray)
        # Divide the LBP image into grids and compute the histogram for each grid.
        h, w = lbp_image.shape
        grid_h = h // self.grid_y
        grid_w = w // self.grid_x
        features = []
        for i in range(self.grid_y):
            for j in range(self.grid_x):
                start_y = i * grid_h
                end_y = (i+1)*grid_h if i < self.grid_y - 1 else h
                start_x = j * grid_w
                end_x = (j+1)*grid_w if j < self.grid_x - 1 else w
                cell = lbp_image[start_y:end_y, start_x:end_x]
                # Compute histogram with 256 bins.
                hist, _ = np.histogram(cell.ravel(), bins=256, range=(0, 256))
                features.extend(hist)
        features = np.array(features, dtype=np.float32)
        # Normalize the feature vector (L2 norm)
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        return features

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    return image

def compute_chi_square_distance(hist1, hist2, eps=1e-10):
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))

def parse_pairs_csv(csv_path, lfw_root):
    df = pd.read_csv(csv_path)
    pairs = []
    for index, row in df.iterrows():
        non_nan = row.dropna()
        if len(non_nan) == 3:
            name = str(non_nan.iloc[0]).strip()
            imagenum1 = int(non_nan.iloc[1])
            imagenum2 = int(non_nan.iloc[2])
            img1 = os.path.join(lfw_root, name, f"{name}_{str(imagenum1).zfill(4)}.jpg")
            img2 = os.path.join(lfw_root, name, f"{name}_{str(imagenum2).zfill(4)}.jpg")
            label = 1
            pairs.append((img1, img2, label))
        elif len(non_nan) == 4:
            name1 = str(non_nan.iloc[0]).strip()
            imagenum1 = int(non_nan.iloc[1])
            name2 = str(non_nan.iloc[2]).strip()
            imagenum2 = int(non_nan.iloc[3])
            img1 = os.path.join(lfw_root, name1, f"{name1}_{str(imagenum1).zfill(4)}.jpg")
            img2 = os.path.join(lfw_root, name2, f"{name2}_{str(imagenum2).zfill(4)}.jpg")
            label = 0
            pairs.append((img1, img2, label))
        else:
            print(f"Unexpected format in row {index}: {row}")
    return pairs

def plot_distances(distances, labels, threshold):
    pos_d = [d for d, l in zip(distances, labels) if l == 1]
    neg_d = [d for d, l in zip(distances, labels) if l == 0]
    plt.figure(figsize=(8, 4))
    plt.hist(pos_d, bins=50, alpha=0.6, label="Same Person")
    plt.hist(neg_d, bins=50, alpha=0.6, label="Different Person")
    plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.3f}")
    plt.xlabel("Chi-square Distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of LBPH Distances")
    plt.legend()
    plt.tight_layout()
    plt.show()

def find_best_threshold(distances, labels):
    thresholds = np.linspace(min(distances), max(distances), 100)
    best_acc = 0
    best_thresh = 0
    for t in thresholds:
        preds = [1 if d < t else 0 for d in distances]
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh, best_acc

def validate_lbph(recognizer, pairs, threshold=None):
    y_true = []
    y_pred = []
    distances = []
    used_threshold = threshold if threshold is not None else recognizer.recognition_threshold

    for idx, (img1_path, img2_path, label) in enumerate(pairs):
        try:
            img1 = load_image(img1_path)
            img2 = load_image(img2_path)
            feat1 = recognizer.extract_features(img1)
            feat2 = recognizer.extract_features(img2)
            if feat1 is None or feat2 is None:
                print(f"Skipping pair {idx} due to face detection failure.")
                continue
            distance = compute_chi_square_distance(feat1, feat2)
            pred = 1 if distance < used_threshold else 0
            y_true.append(label)
            y_pred.append(pred)
            distances.append(distance)
            print(f"Pair {idx}: {os.path.basename(img1_path)} vs {os.path.basename(img2_path)} | Distance = {distance:.3f} | Label = {label} | Prediction = {pred}")
        except Exception as e:
            print(f"Error in pair {idx}: {e}")
            continue

    accuracy_val = accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, [-d for d in distances])
    except Exception as e:
        print(f"Error computing ROC AUC: {e}")
        roc_auc = None
    conf_matrix = confusion_matrix(y_true, y_pred)
    metrics = {
        'accuracy': accuracy_val,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'distances': distances,
        'y_true': y_true
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="LFW Validation for LBPH Face Recognizer")
    parser.add_argument("--pairs_csv", type=str, default="dataset/lfw/pairs.csv", help="Path to CSV file with image pairs")
    parser.add_argument("--lfw_root", type=str, default="dataset/lfw/lfw-deepfunneled", help="Root directory for LFW images")
    parser.add_argument("--threshold", type=float, default=13.0, help="Initial LBPH distance threshold")
    parser.add_argument("--grid_x", type=int, default=8, help="Number of grid columns for LBPH")
    parser.add_argument("--grid_y", type=int, default=8, help="Number of grid rows for LBPH")
    args = parser.parse_args()

    recognizer = FaceRecognizerLBPH(recognition_threshold=args.threshold, grid_x=args.grid_x, grid_y=args.grid_y)
    pairs = parse_pairs_csv(args.pairs_csv, args.lfw_root)
    metrics = validate_lbph(recognizer, pairs, threshold=args.threshold)

    print("\nValidation Results:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    if metrics['roc_auc'] is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])

    plot_distances(metrics['distances'], metrics['y_true'], args.threshold)
    best_t, best_acc = find_best_threshold(metrics['distances'], metrics['y_true'])
    print(f"\nOptimal threshold found: {best_t:.4f} with Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
