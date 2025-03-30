import cv2
import numpy as np
import pickle
import os
import time
import collections
import psutil
import GPUtil

# Define model filename and LBPH threshold
MODEL_FILENAME = "trained/lbph_model.pkl"
RECOGNITION_THRESHOLD = 30

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
# Face Recognizer using LBPH
# =============================================================================
class FaceRecognizerLBPH:
    def __init__(self, recognition_threshold=RECOGNITION_THRESHOLD):
        self.recognition_threshold = recognition_threshold
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.training_images = []  # List of grayscale face images
        self.training_labels = []  # List of integer labels corresponding to training images
        self.label_to_int = {}     # Maps string labels to integer IDs
        self.int_to_label = {}     # Maps integer IDs back to string labels
        self.next_label = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpickleable LBPH model from the state.
        state['model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = cv2.face.LBPHFaceRecognizer_create()
        if self.training_images and self.training_labels:
            self.model.train(self.training_images, np.array(self.training_labels))

    def add_training_sample(self, face_image, label):
        # Convert the face image to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        # If label is new, assign a new integer ID.
        if label not in self.label_to_int:
            self.label_to_int[label] = self.next_label
            self.int_to_label[self.next_label] = label
            self.next_label += 1
        int_label = self.label_to_int[label]
        self.training_images.append(gray)
        self.training_labels.append(int_label)
        # Re-train the recognizer with all available training samples.
        self.model.train(self.training_images, np.array(self.training_labels))

    def recognize(self, face_image):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        if len(self.training_images) == 0:
            return "Unknown", float('inf')
        # The predict method returns a tuple: (predicted_label, confidence)
        int_label, confidence = self.model.predict(gray)
        if confidence < self.recognition_threshold:
            return self.int_to_label.get(int_label, "Unknown"), confidence
        else:
            return "Unknown", confidence

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

    face_detector = FaceDetectorHaar()
    face_recognizer = load_model() or FaceRecognizerLBPH()

    training_mode = False
    target_training_samples = 250  # Adjust as needed.
    current_label = None
    training_sample_count = 0

    print("Press 'T' to start training mode with a new label. Press 'q' to quit.")

    # Initialize system metrics
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
        """
        if frame_count % 10 == 0:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_load = gpus[0].load * 100.0
                gpu_buffer.append(gpu_load)
                avg_gpu = sum(gpu_buffer) / len(gpu_buffer)
            else:
                avg_gpu = 0.0
        """
        avg_gpu= 0.0
        faces = face_detector.detect(frame)

        if training_mode:
            if faces is not None and len(faces) > 0:
                x, y, w, h = faces[0][0:4].astype(int)
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue
                # Measure inference time for training sample extraction
                start_inf = time.time()
                # Change size of ROI for better accuracy
                face_roi_resized = cv2.resize(face_roi, (256, 256))
                face_recognizer.add_training_sample(face_roi_resized, current_label)
                end_inf = time.time()
                inf_time = (end_inf - start_inf) * 1000  # in milliseconds
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
                print(f"LBPH model updated with label '{current_label}'. Returning to recognition mode.")

        else:
            if len(face_recognizer.training_images) == 0:
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
                        label, confidence = face_recognizer.recognize(face_roi)
                        end_inf = time.time()
                        inf_time = (end_inf - start_inf) * 1000  # in milliseconds
                        inf_time_buffer.append(inf_time)
                        avg_inf_time = sum(inf_time_buffer) / len(inf_time_buffer)
                        if confidence < face_recognizer.recognition_threshold:
                            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, f"Unknown ({confidence:.2f})", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    except Exception as e:
                        cv2.putText(frame, "Recognition error", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # -------------------------
        # Overlay Metrics on Frame (Top-Right Corner)
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
