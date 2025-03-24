import cv2
import numpy as np
import pickle
import os

MODEL_FILENAME = "trained/arcface_model.pkl"  # Path for saving/loading the recognizer state
RECOGNITION_THRESHOLD = 0.3  # Cosine distance threshold (adjust based on empirical results)

# =============================================================================
# Face Detector using YuNet (OpenCV implementation)
# =============================================================================
class FaceDetectorYunet:
    def __init__(self, model_path="../detection/models/face_detection_yunet_2023mar.onnx", input_size=(0, 0),
                 score_threshold=0.5):
        self.detector = cv2.FaceDetectorYN_create(model_path, "", input_size, score_threshold)
        self.input_size = input_size

    def detect(self, image):
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return faces

# =============================================================================
# Face Recognizer using ArcFace
# =============================================================================
class FaceRecognizerArcFace:
    def __init__(self, model_path="models/arcfaceresnet100-8.onnx", recognition_threshold=RECOGNITION_THRESHOLD):
        self.model_path = model_path
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.features_database = []  # List of normalized embeddings for known faces.
        self.labels = []             # Corresponding labels.
        self.recognition_threshold = recognition_threshold

    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.model is None:
            self.model = cv2.dnn.readNetFromONNX(self.model_path)

    def preprocess(self, face_image):
        """
        Preprocesses the face image for ArcFace.
        Converts BGR to RGB, resizes to 112x112,
        and normalizes pixel values to the range [-1, 1] by using:
            output = (image - 127.5) / 128
        which is a common preprocessing for ArcFace models.
        """
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (112, 112))
        blob = cv2.dnn.blobFromImage(face_resized)
        return blob

    def extract_features(self, face_image):
        blob = self.preprocess(face_image)
        self.model.setInput(blob)
        embedding = self.model.forward()  # Expected shape: (1, embedding_size)
        embedding = embedding.flatten()
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
        similarities = np.array([np.dot(embedding, feat) for feat in self.features_database])
        distances = 1 - similarities  # Cosine distance for normalized vectors.
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        recognized_label = self.labels[min_index]
        return recognized_label, min_distance

# =============================================================================
# Utility functions for saving and loading the model state.
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
# Main script: live camera face detection, training, and recognition pipeline
# =============================================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    face_detector = FaceDetectorYunet()
    face_recognizer = load_model() or FaceRecognizerArcFace()

    training_mode = False
    target_training_samples = 250  # Adjust as needed.
    current_label = None
    training_sample_count = 0

    print("Press 'T' to start training mode with a new label. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector.detect(frame)

        if training_mode:
            if faces is not None and len(faces) > 0:
                x, y, w, h = faces[0][0:4].astype(int)
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue
                # Add training sample
                face_recognizer.add_training_sample(face_roi, current_label)
                training_sample_count += 1
                cv2.putText(frame, f"Training {current_label}: {training_sample_count}/{target_training_samples}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if training_sample_count >= target_training_samples:
                training_mode = False
                training_sample_count = 0
                save_model(face_recognizer)
                print(f"ArcFace model updated with label '{current_label}'. Returning to recognition mode.")

        else:
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
                        label, distance = face_recognizer.recognize(face_roi)
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