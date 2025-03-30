import os
import cv2
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

class FaceRecognizerArcFace:
    def __init__(self, model_path="models/arcfaceresnet100-8.onnx", recognition_threshold=0.3):
        self.model_path = model_path
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.recognition_threshold = recognition_threshold
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def preprocess(self, face_image):
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (112, 112))
        blob = cv2.dnn.blobFromImage(face_resized)
        return blob

    def detect_and_crop_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        return face

    def extract_features(self, image):
        face = self.detect_and_crop_face(image)
        if face is None:
            return None
        blob = self.preprocess(face)
        self.model.setInput(blob)
        embedding = self.model.forward().flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    return image

def compute_cosine_distance(emb1, emb2):
    similarity = np.dot(emb1, emb2)
    return 1 - similarity

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
            print(f"Fila {index} con formato inesperado: {row}")
    return pairs

def plot_distances(distances, labels, threshold):
    pos_d = [d for d, l in zip(distances, labels) if l == 1]
    neg_d = [d for d, l in zip(distances, labels) if l == 0]
    plt.figure(figsize=(8, 4))
    plt.hist(pos_d, bins=50, alpha=0.6, label="Misma persona")
    plt.hist(neg_d, bins=50, alpha=0.6, label="Distinta persona")
    plt.axvline(threshold, color='red', linestyle='--', label=f"Umbral = {threshold:.3f}")
    plt.xlabel("Distancia coseno")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de distancias")
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

def validate_arcface(recognizer, pairs, threshold=None):
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
                print(f"Omitiendo par {idx} por falta de detección facial.")
                continue
            distance = compute_cosine_distance(emb1, emb2)
            pred = 1 if distance < used_threshold else 0
            y_true.append(label)
            y_pred.append(pred)
            distances.append(distance)
            print(f"Par {idx}: {os.path.basename(img1_path)} vs {os.path.basename(img2_path)} | Distancia = {distance:.3f} | Label = {label} | Prediccion = {pred}")
        except Exception as e:
            print(f"Error en par {idx}: {e}")
            continue

    accuracy_val = accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, [-d for d in distances])
    except Exception as e:
        print(f"Error al calcular ROC AUC: {e}")
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
    parser = argparse.ArgumentParser(description="Validacion LFW para ArcFace con preprocesamiento correcto")
    parser.add_argument("--pairs_csv", type=str, default="dataset/lfw/pairs.csv", help="Ruta al archivo CSV con pares de imágenes")
    parser.add_argument("--lfw_root", type=str, default="dataset/lfw/lfw-deepfunneled", help="Directorio raíz de las imágenes LFW funneled")
    parser.add_argument("--model_path", type=str, default="models/arcfaceresnet100-8.onnx", help="Ruta al modelo ONNX")
    parser.add_argument("--threshold", type=float, default=0.6, help="Umbral inicial de distancia coseno")
    args = parser.parse_args()

    recognizer = FaceRecognizerArcFace(model_path=args.model_path, recognition_threshold=args.threshold)
    pairs = parse_pairs_csv(args.pairs_csv, args.lfw_root)
    metrics = validate_arcface(recognizer, pairs, threshold=args.threshold)

    print("\nResultados de la validacion:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    if metrics['roc_auc'] is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print("Matriz de confusion:")
    print(metrics['confusion_matrix'])

    plot_distances(metrics['distances'], metrics['y_true'], args.threshold)
    best_t, best_acc = find_best_threshold(metrics['distances'], metrics['y_true'])
    print(f"\nUmbral optimo encontrado: {best_t:.4f} con Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
