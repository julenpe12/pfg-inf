import cv2
import numpy as np
import pickle
import os
import time
import collections
import psutil
import GPUtil
import torch
import torchvision.transforms as transforms
from PIL import Image

MODEL_FILENAME = "trained/mobilefacenet_model.pkl"  # File to save/load recognizer state
RECOGNITION_THRESHOLD = 0.3  # Cosine distance threshold

# =============================================================================
# Face Detector using Haar Cascade
# =============================================================================
class FaceDetectorHaar:
    def __init__(self, cascade_path=cv2.data.haarcascades + "haarcascade_frontalface_default.xml", scaleFactor=1.1, minNeighbors=5):
        if not os.path.exists(cascade_path):
            raise ValueError(f"Cascade file not found: {cascade_path}")
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors)
        return faces

# =============================================================================
# Face Recognizer using MobileFaceNet TorchScript
# =============================================================================
class FaceRecognizerMobileFaceNetTorch:
    def __init__(self, model_path="models/mobilefacenet_scripted.pt", recognition_threshold=RECOGNITION_THRESHOLD, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        # Load the TorchScript MobileFaceNet model.
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.features_database = []  # List to store normalized embeddings.
        self.labels = []             # Corresponding labels for the embeddings.
        self.recognition_threshold = recognition_threshold

        # Define preprocessing: MobileFaceNet expects 112x112 images.
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-picklable model from the state.
        state['model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reload the model if missing.
        if self.model is None:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()

    def preprocess(self, face_image):
        # Convert face ROI (numpy array) to a PIL image.
        pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_image).unsqueeze(0)  # Add batch dimension.
        return tensor.to(self.device)

    def extract_features(self, face_image):
        tensor = self.preprocess(face_image)
        with torch.no_grad():
            embedding = self.model(tensor)
        embedding = embedding.flatten().cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def add_training_sample(self, face_image, label):
        embedding = self.extract_features(face_image)
        self.features_database.append(embedding)
        self.labels.append(label)

    def recognize(self, face_image):
        embedding = self.extract_features(face_image)
        if len(self.features_database) == 0:
            return "Unknown", 1.0
        # Compute cosine similarity (distance = 1 - dot product for normalized vectors).
        similarities = np.array([np.dot(embedding, feat) for feat in self.features_database])
        distances = 1 - similarities
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        recognized_label = self.labels[min_index]
        return recognized_label, min_distance

# =============================================================================
# Utility functions to save/load the recognizer state.
# =============================================================================
def save_model(recognizer, filename=MODEL_FILENAME):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(recognizer, f)
    print("Model state saved to", filename)

def load_model(filename=MODEL_FILENAME):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            recognizer = pickle.load(f)
        print("Model state loaded from", filename)
        return recognizer
    else:
        return None

# =============================================================================
# Main script: live camera face detection, training, and recognition
# =============================================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    face_detector = FaceDetectorHaar()
    face_recognizer = load_model() or FaceRecognizerMobileFaceNetTorch()

    training_mode = False
    target_training_samples = 250  # Number of samples to collect per label.
    current_label = None
    training_sample_count = 0

    print("Press 'T' to start training mode with a new label. Press 'q' to quit.")

    # Initialize system metrics.
    process = psutil.Process(os.getpid())
    logical_cores = psutil.cpu_count(logical=True)
    prev_frame_time = time.time()
    fps_buffer = collections.deque(maxlen=30)
    cpu_buffer = collections.deque(maxlen=30)
    mem_buffer = collections.deque(maxlen=30)
    gpu_buffer = collections.deque(maxlen=30)
    inf_time_buffer = collections.deque(maxlen=30)
    frame_count = 0
    avg_gpu = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------
        # System Metrics Calculations
        # -------------------------
        current_time = time.time()
        fps = 1.0 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        fps_buffer.append(fps)
        avg_fps = sum(fps_buffer) / len(fps_buffer)

        cpu_usage = process.cpu_percent(interval=None)
        norm_cpu = cpu_usage / logical_cores
        cpu_buffer.append(norm_cpu)
        avg_cpu = sum(cpu_buffer) / len(cpu_buffer)

        mem_usage = process.memory_info().rss / (1024 * 1024)
        mem_buffer.append(mem_usage)
        avg_mem = sum(mem_buffer) / len(mem_buffer)

        frame_count += 1
        if frame_count % 10 == 0:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_load = gpus[0].load * 100.0
                gpu_buffer.append(gpu_load)
                avg_gpu = sum(gpu_buffer) / len(gpu_buffer)
            else:
                avg_gpu = 0.0

        # -------------------------
        # Face Detection
        # -------------------------
        faces = face_detector.detect(frame)

        # -------------------------
        # Training Mode: Collect embeddings with label.
        # -------------------------
        if training_mode:
            if faces is not None and len(faces) > 0:
                x, y, w, h = faces[0][0:4].astype(int)
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue
                start_inf = time.time()
                face_recognizer.add_training_sample(face_roi, current_label)
                end_inf = time.time()
                inf_time = (end_inf - start_inf) * 1000  # Inference time in ms.
                inf_time_buffer.append(inf_time)
                avg_inf_time = sum(inf_time_buffer) / len(inf_time_buffer)

                cv2.putText(frame, f"Training {current_label}: {training_sample_count+1}/{target_training_samples}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                training_sample_count += 1

            if training_sample_count >= target_training_samples:
                training_mode = False
                training_sample_count = 0
                save_model(face_recognizer)
                print(f"MobileFaceNet model updated with label '{current_label}'. Returning to recognition mode.")

        # -------------------------
        # Recognition Mode
        # -------------------------
        else:
            avg_inf_time = 0
            if len(face_recognizer.features_database) == 0:
                cv2.putText(frame, "No training data", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if faces is not None:
                for face in faces:
                    x, y, w, h = face[0:4].astype(int)
                    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                        continue
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size == 0:
                        continue
                    try:
                        start_inf = time.time()
                        label, distance = face_recognizer.recognize(face_roi)
                        end_inf = time.time()
                        inf_time = (end_inf - start_inf) * 1000
                        inf_time_buffer.append(inf_time)
                        avg_inf_time = sum(inf_time_buffer) / len(inf_time_buffer)
                        if distance < face_recognizer.recognition_threshold:
                            cv2.putText(frame, f"{label} ({distance:.2f})", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, f"Unknown ({distance:.2f})", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    except Exception as e:
                        cv2.putText(frame, "Recognition error", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # -------------------------
        # Overlay Metrics on Frame
        # -------------------------
        text_x = 7
        cv2.putText(frame, f"FPS: {int(avg_fps)}", (text_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Inf: {int(avg_inf_time)} ms", (text_x, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"CPU: {avg_cpu:.1f}% per core", (text_x, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Mem: {avg_mem:.2f} MB", (text_x, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"GPU: {avg_gpu:.1f}%", (text_x, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key in [ord('T'), ord('t')]:
            if not training_mode:
                current_label = input("Enter label for training: ").strip()
                if current_label == "":
                    print("Empty label. Training mode not activated.")
                else:
                    training_mode = True
                    training_sample_count = 0
                    print(f"Training mode activated for label '{current_label}'. Collecting training images...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
