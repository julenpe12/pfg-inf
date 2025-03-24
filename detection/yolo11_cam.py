import cv2
import time
import collections
import psutil
import os
from ultralytics import YOLO

def main():
    # Load the YOLOv11 model weights
    model_path = "models/yolov11l-face.pt"  # Replace with your actual model file
    model = YOLO(model_path)
    # GPU
    model.to("cuda")
    
    # CPU
    #model.to("cpu")
    
    # Open the default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    process = psutil.Process(os.getpid())
    logical_cores = psutil.cpu_count(logical=True)
    
    # Initialize buffers for FPS, inference time, normalized CPU load, and memory usage
    prev_frame_time = time.time()
    fps_buffer = collections.deque(maxlen=30)
    inf_time_buffer = collections.deque(maxlen=30)
    cpu_buffer = collections.deque(maxlen=30)
    mem_buffer = collections.deque(maxlen=30)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        mem_usage = process.memory_info().rss / (1024 * 1024)
        mem_buffer.append(mem_usage)
        avg_mem = sum(mem_buffer) / len(mem_buffer)
        
        current_time = time.time()
        fps = 1.0 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        fps_buffer.append(fps)
        avg_fps = sum(fps_buffer) / len(fps_buffer)
        
        cpu_usage = process.cpu_percent(interval=None)
        norm_cpu = cpu_usage / logical_cores
        cpu_buffer.append(norm_cpu)
        avg_cpu = sum(cpu_buffer) / len(cpu_buffer)
        
        start_inf = time.time()
        results = model(frame)
        end_inf = time.time()
        inf_time = (end_inf - start_inf) * 1000  # in ms
        inf_time_buffer.append(inf_time)
        avg_inf_time = sum(inf_time_buffer) / len(inf_time_buffer)
        
        preds = results[0]  # First batch prediction
        for box in preds.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{score:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        cv2.putText(frame, f"FPS: {int(avg_fps)}", (7, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Inf: {int(avg_inf_time)} ms", (7, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"CPU: {avg_cpu:.1f}% per core", (7, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Mem: {avg_mem:.2f} MB", (7, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Cores: {logical_cores}", (7, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,255,0), 2, cv2.LINE_AA)
        
        cv2.imshow("YOLOv11-L - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
