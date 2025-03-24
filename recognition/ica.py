import cv2
import numpy as np
from sklearn.decomposition import FastICA
import pickle
import os

MODEL_FILENAME = "trained/ica_model.pkl"  # Path for saving/loading the trained model state
RECOGNITION_THRESHOLD = 10 # Adjust based on empirical observation of distances

# =============================================================================
# Face Detector using YuNet (OpenCV implementation)
# =============================================================================
class FaceDetectorYunet:
    def __init__(self, model_path="../detection/models/face_detection_yunet_2023mar.onnx", input_size=(0, 0),
                 score_threshold=0.5):
        """
        Initializes the YuNet face detector.
        """
        self.detector = cv2.FaceDetectorYN_create(model_path, "", input_size, score_threshold)
        self.input_size = input_size

    def detect(self, image):
        """
        Detects faces in the input image.
        Returns a NumPy array of detections where each row contains
        [x, y, width, height, ...].
        """
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return faces

# =============================================================================
# Face Recognizer using Independent Component Analysis (ICA)
# =============================================================================
class FaceRecognizerICA:
    def __init__(self, n_components=None):
        """
        Initializes the ICA-based face recognizer.
        """
        self.ica = FastICA(n_components=n_components)
        self.trained = False
        self.features_database = []  # Transformed feature vectors for known faces.
        self.labels = []             # Associated labels for each known face.
        # Store all raw training samples (flattened images) and their labels.
        self.training_data = []
        self.training_labels = []

    def train(self, new_face_images, new_labels):
        """
        Trains (or retrains) the ICA model using new training samples.
        New samples are appended to the existing training set, and the ICA model
        is re-trained on the entire dataset.
        """
        self.training_data.extend(new_face_images)
        self.training_labels.extend(new_labels)
        X = np.array(self.training_data)
        self.ica.fit(X)
        self.trained = True
        features = self.ica.transform(X)
        self.features_database = features
        self.labels = self.training_labels

    def extract_features(self, face_image):
        """
        Extracts a feature vector from a preprocessed face image.
        The face image must be preprocessed (flattened, resized, normalized)
        in the same way as during training.
        """
        face_vector = face_image.flatten().astype(np.float32)
        if not self.trained:
            raise ValueError("ICA model is not trained. Please train the model with known face images.")
        features = self.ica.transform([face_vector])
        return features[0]

    def recognize(self, face_image):
        """
        Recognizes a face by comparing its ICA features with the database.
        Returns the label of the closest match and the Euclidean distance.
        """
        features = self.extract_features(face_image)
        distances = np.linalg.norm(self.features_database - features, axis=1)
        min_index = np.argmin(distances)
        recognized_label = self.labels[min_index]
        return recognized_label, distances[min_index]

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

    # Initialize face detector.
    face_detector = FaceDetectorYunet()
    # Load previously saved recognizer if available; otherwise, create a new one.
    face_recognizer = load_model() or FaceRecognizerICA(n_components=50)

    training_mode = False
    training_samples = []    # New training samples for current label.
    target_training_samples = 500  # Number of training samples to collect per label.
    current_label = None

    print("Press 'T' to start training mode with a new label. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector.detect(frame)

        if training_mode:
            if faces is not None and len(faces) > 0:
                # Use the first detected face for training.
                x, y, w, h = faces[0][0:4].astype(int)
                # Ensure ROI is within frame boundaries.
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue
                face_roi = frame[y:y+h, x:x+w]
                # Check that ROI is not empty.
                if face_roi.size == 0:
                    continue
                try:
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                except cv2.error as e:
                    print("Error converting face ROI to grayscale:", e)
                    continue
                face_resized = cv2.resize(face_gray, (100, 100))
                training_samples.append(face_resized.flatten().astype(np.float32))
                cv2.putText(frame, f"Training {current_label}: {len(training_samples)}/{target_training_samples}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Once enough training samples are collected, train the model.
            if len(training_samples) >= target_training_samples:
                new_labels = [current_label] * len(training_samples)
                face_recognizer.train(training_samples, new_labels)
                training_mode = False
                training_samples = []  # Reset new training samples.
                save_model(face_recognizer)
                print(f"ICA model trained with label '{current_label}'. Returning to recognition mode.")

        else:
            if not face_recognizer.trained:
                cv2.putText(frame, "ICA not trained", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if faces is not None:
                for face in faces:
                    x, y, w, h = face[0:4].astype(int)
                    # Ensure ROI is within frame boundaries.
                    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                        continue
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size == 0:
                        continue
                    try:
                        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    except cv2.error as e:
                        print("Error converting face ROI to grayscale:", e)
                        continue
                    face_resized = cv2.resize(face_gray, (100, 100))
                    if face_recognizer.trained:
                        try:
                            label, distance = face_recognizer.recognize(face_resized)
                            # Use the RECOGNITION_THRESHOLD to decide between known and unknown.
                            if distance < RECOGNITION_THRESHOLD:
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

        # Exit if 'q' is pressed.
        if key == ord('q'):
            break
        # Activate training mode when 'T' (or 't') is pressed.
        if key in [ord('T'), ord('t')]:
            if not training_mode:
                current_label = input("Enter label for training: ").strip()
                if current_label == "":
                    print("Empty label. Training mode not activated.")
                else:
                    training_mode = True
                    training_samples = []
                    print(f"Training mode activated for label '{current_label}'. Collecting training images...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()