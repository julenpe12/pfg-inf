import cv2
import time
import collections
import psutil
import os
import numpy as np

def main():
    # Path to YuNet ONNX model
    model_path = "models/face_detection_yunet_2023mar_int8bq.onnx"
    
    # Create YuNet face detector with a score threshold of 0.5
    face_detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (0, 0),
        0.5
    )
    
    # Open the default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    process = psutil.Process(os.getpid())
    logical_cores = psutil.cpu_count(logical=True)
    
    # Initialize buffers for FPS, inference time, CPU load, and memory usage
    prev_frame_time = time.time()
    fps_buffer = collections.deque(maxlen=30)
    inf_time_buffer = collections.deque(maxlen=30)
    cpu_buffer = collections.deque(maxlen=30)
    mem_buffer = collections.deque(maxlen=30)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        
        # Update memory usage
        mem_usage = process.memory_info().rss / (1024 * 1024)
        mem_buffer.append(mem_usage)
        avg_mem = sum(mem_buffer) / len(mem_buffer)
        
        # Update FPS calculation
        current_time = time.time()
        fps = 1.0 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        fps_buffer.append(fps)
        avg_fps = sum(fps_buffer) / len(fps_buffer)
        
        # Update CPU usage
        cpu_usage = process.cpu_percent(interval=None)
        norm_cpu = cpu_usage / logical_cores
        cpu_buffer.append(norm_cpu)
        avg_cpu = sum(cpu_buffer) / len(cpu_buffer)
        
        # Update input size in case frame dimensions change
        h, w = frame.shape[:2]
        face_detector.setInputSize((w, h))
        
        # Perform face detection and measure inference time
        start_inf = time.time()
        rc, faces = face_detector.detect(frame)
        end_inf = time.time()
        inf_time = (end_inf - start_inf) * 1000  # Inference time in milliseconds
        inf_time_buffer.append(inf_time)
        avg_inf_time = sum(inf_time_buffer) / len(inf_time_buffer)
        
        # Detailed processing of detected faces
        if rc > 0 and faces is not None:
            for face in faces:
                # Extract the bounding box: first 4 values (x, y, width, height)
                x, y, box_w, box_h = face[:4].astype(int)
                # Extract the 5 facial landmarks: next 10 values (5 pairs of x,y coordinates)
                landmarks = face[4:14].reshape((5, 2)).astype(int)
                # Extract the confidence score: last value
                score = face[-1]
                
                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                # Draw each landmark as a small circle
                for (lx, ly) in landmarks:
                    cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)
                # Annotate the detection with the confidence score
                cv2.putText(frame, f"{score:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display performance metrics and additional information on the frame
        cv2.putText(frame, f"FPS: {int(avg_fps)}", (7, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Inf: {int(avg_inf_time)} ms", (7, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"CPU: {avg_cpu:.1f}% per core", (7, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Mem: {avg_mem:.2f} MB", (7, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Cores: {logical_cores}", (7, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2, cv2.LINE_AA)
        # Display the number of detected faces
        cv2.putText(frame, f"Faces: {rc}", (7, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("YuNet Face Detection - Detailed Utilization (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()