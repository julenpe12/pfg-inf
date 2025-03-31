import os
import cv2
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import torch

class FaceRecognizerMobileFaceNet:
    def __init__(self, model_path="models/mobilefacenet_scripted.pt", recognition_threshold=0.6, device=None):
        self.model_path = model_path
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the PyTorch MobileFaceNet model and set it to evaluation mode.
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        # Haar Cascade for face detection (used for cropping)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognition_threshold = recognition_threshold

    def preprocess(self, face_image):
        """
        Preprocess the face image for MobileFaceNet.
        - Convert from BGR to RGB.
        - Resize to 112x112.
        - Scale pixel values to [0, 1].
        - Convert to PyTorch tensor with shape (1, 3, 112, 112).
        """
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (112, 112))
        face_np = face_resized.astype(np.float32) / 255.0  # Scale to [0, 1]
        # Change from HWC to CHW layout
        face_np = np.transpose(face_np, (2, 0, 1))
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(face_np).unsqueeze(0).to(self.device)
        return tensor

    def detect_and_crop_face(self, image):
        """
        Detects a face using Haar cascade and returns the cropped face.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        return face

    def extract_features(self, image):
        """
        Detects and crops the face from the input image, preprocesses it,
        runs inference through MobileFaceNet, and returns embedding.
        """
        face = self.detect_and_crop_face(image)
        if face is None:
            return None
        tensor = self.preprocess(face)
        with torch.no_grad():
            embedding = self.model(tensor)
        embedding = embedding.squeeze(0).cpu().numpy()
        return embedding

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    return image

def compute_distance(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)


def parse_pairs_csv(csv_path, lfw_root):
    df = pd.read_csv(csv_path)
    pairs = []
    for idx, row in df.iterrows():
        non_nan = row.dropna()
        if len(non_nan) == 3:
            # Positive pair: same person with two images.
            name = str(non_nan.iloc[0]).strip()
            imgnum1 = int(non_nan.iloc[1])
            imgnum2 = int(non_nan.iloc[2])
            img1 = os.path.join(lfw_root, name, f"{name}_{str(imgnum1).zfill(4)}.jpg")
            img2 = os.path.join(lfw_root, name, f"{name}_{str(imgnum2).zfill(4)}.jpg")
            label = 1
            pairs.append((img1, img2, label))
        elif len(non_nan) == 4:
            # Negative pair: different persons.
            name1 = str(non_nan.iloc[0]).strip()
            imgnum1 = int(non_nan.iloc[1])
            name2 = str(non_nan.iloc[2]).strip()
            imgnum2 = int(non_nan.iloc[3])
            img1 = os.path.join(lfw_root, name1, f"{name1}_{str(imgnum1).zfill(4)}.jpg")
            img2 = os.path.join(lfw_root, name2, f"{name2}_{str(imgnum2).zfill(4)}.jpg")
            label = 0
            pairs.append((img1, img2, label))
        else:
            print(f"Row {idx} has an unexpected format: {row}")
    return pairs

def plot_distances(distances, labels, threshold):
    pos_d = [d for d, l in zip(distances, labels) if l == 1]
    neg_d = [d for d, l in zip(distances, labels) if l == 0]
    plt.figure(figsize=(8, 4))
    plt.hist(pos_d, bins=50, alpha=0.6, label="Same person")
    plt.hist(neg_d, bins=50, alpha=0.6, label="Different persons")
    plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.3f}")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Frequency")
    plt.title("Distance Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

def find_best_threshold(distances, labels):
    thresholds = np.linspace(0.2, 0.8, 100)
    best_acc = 0
    best_thresh = 0
    for t in thresholds:
        preds = [1 if d < t else 0 for d in distances]
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh, best_acc

def validate_mobilefacenet(recognizer, pairs, threshold=None):
    y_true = []
    y_pred = []
    distances = []
    used_threshold = threshold if threshold is not None else recognizer.recognition_threshold

    for idx, (img1_path, img2_path, label) in enumerate(pairs):
        try:
            img1 = load_image(img1_path)
            img2 = load_image(img2_path)
            emb1 = recognizer.extract_features(img1)
            emb2 = recognizer.extract_features(img2)
            if emb1 is None or emb2 is None:
                print(f"Skipping pair {idx} due to failed face detection.")
                continue
            distance = compute_distance(emb1, emb2)
            pred = 1 if distance < used_threshold else 0
            y_true.append(label)
            y_pred.append(pred)
            distances.append(distance)
            print(f"Pair {idx}: {os.path.basename(img1_path)} vs {os.path.basename(img2_path)} | "
                  f"Distance = {distance:.3f} | Label = {label} | Prediction = {pred}")
        except Exception as e:
            print(f"Error in pair {idx}: {e}")
            continue

    accuracy_val = accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, [-d for d in distances])
    except Exception as e:
        print(f"Error calculating ROC AUC: {e}")
        roc_auc = None
    conf_matrix = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": accuracy_val,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix,
        "distances": distances,
        "y_true": y_true
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="LFW Validation for MobileFaceNet (.pt)")
    parser.add_argument("--pairs_csv", type=str, default="dataset/lfw/pairs.csv",
                        help="Path to the CSV file with image pairs")
    parser.add_argument("--lfw_root", type=str, default="dataset/lfw/lfw-deepfunneled",
                        help="Root directory of LFW images")
    parser.add_argument("--model_path", type=str, default="models/mobilefacenet_scripted.pt",
                        help="Path to the MobileFaceNet .pt model")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Euclidean distance (L2) threshold")
    args = parser.parse_args()

    recognizer = FaceRecognizerMobileFaceNet(model_path=args.model_path, recognition_threshold=args.threshold)
    pairs = parse_pairs_csv(args.pairs_csv, args.lfw_root)
    metrics = validate_mobilefacenet(recognizer, pairs, threshold=args.threshold)
    
    print("\nValidation Results:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    if metrics["roc_auc"] is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    plot_distances(metrics["distances"], metrics["y_true"], args.threshold)
    best_thresh, best_acc = find_best_threshold(metrics["distances"], metrics["y_true"])
    print(f"\nOptimal threshold found: {best_thresh:.4f} with Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
