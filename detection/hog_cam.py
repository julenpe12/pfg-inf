import cv2
import dlib
import time
import collections
import psutil
import os

def main():
    # Initialize dlib's HOG-based face detector
    detector = dlib.get_frontal_face_detector()
    
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
        
        cpu_usage = process.cpu_percent()
        norm_cpu = cpu_usage / logical_cores
        cpu_buffer.append(norm_cpu)
        avg_cpu = sum(cpu_buffer) / len(cpu_buffer)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start_inf = time.time()
        faces = detector(gray, 1)
        end_inf = time.time()
        inf_time = (end_inf - start_inf) * 1000  # in ms
        inf_time_buffer.append(inf_time)
        avg_inf_time = sum(inf_time_buffer) / len(inf_time_buffer)
        
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
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
        
        cv2.imshow("HOG Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()