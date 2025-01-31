import cv2
import os
import time
import threading
from queue import Queue
import torch
from ultralytics import YOLO
import datetime

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to YOLO models
model_paths = {
    "gesture": "C:/Users/flutter/Desktop/rstpcamtest/all7modelspt/gesture_(4class)_10_24_24_best.pt",
    "weapon": "C:/Users/flutter/Desktop/rstpcamtest/all7modelspt/weapon_8class_separated_6_12_24_yolov8n_best.pt",
    "pet": "C:/Users/flutter/Desktop/rstpcamtest/all7modelspt/pet_baby5class_22_11_24_yolov11_best.pt",
    "fire": "C:/Users/flutter/Desktop/rstpcamtest/all7modelspt/fire_26_11_24_best.pt",
    "parcel": "C:/Users/flutter/Desktop/rstpcamtest/all7modelspt/parcel_19_12_24_yolov8n_best.pt",
    "combinedmodel": "C:/Users/flutter/Desktop/rstpcamtest/all7modelspt/combinedModel_27_class_9_12_24_best.pt",
    "FALLJUMPCOMBINED": "C:/Users/flutter/Desktop/rstpcamtest/all7modelspt/fall_jump_combined_1_1_25_yolo11n_best.pt",
}

# Load YOLO models
models = {name: YOLO(path).to(device) for name, path in model_paths.items()}

# Print the class names for each model
for model_name, model in models.items():
    print(f"\nClass names for the '{model_name}' model:")
    print(model.names)

# Output directories for detections
output_base_dir = "C:/finalstreamdataset"
os.makedirs(output_base_dir, exist_ok=True)

detection_dirs = {}
confidence_ranges_dirs = {}
confidence_ranges = [
    (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
    (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)
]

for model_name in model_paths.keys():
    model_dir = os.path.join(output_base_dir, f"{model_name}_detected_with_box")
    os.makedirs(model_dir, exist_ok=True)
    detection_dirs[model_name] = model_dir
    confidence_ranges_dirs[model_name] = {}

    for low, high in confidence_ranges:
        range_name = f"{int(low*10)}-{int(high*10)}"
        conf_dir = os.path.join(model_dir, range_name)
        images_dir = os.path.join(conf_dir, "images")
        labels_dir = os.path.join(conf_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        confidence_ranges_dirs[model_name][(low, high)] = {
            "images_dir": images_dir,
            "labels_dir": labels_dir
        }

# Initialize a lock for writing to log files
log_lock = threading.Lock()

# Process frame for a single model
def process_frame(frame, model, confidence_ranges_dirs, classes):
    frame_copy = frame.copy()
    height, width, _ = frame.shape
    results = model(frame, verbose=False)

    for result in results:
        for box, score, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if score > 0.3:  # Process detections with confidence > 0.3
                cls = int(cls)
                x1, y1, x2, y2 = map(int, box)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")

                # Normalize YOLO format coordinates
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height

                for (low, high), dirs in confidence_ranges_dirs.items():
                    if low <= score < high:
                        # Save image
                        image_path = os.path.join(dirs["images_dir"], f"{timestamp}.jpg")
                        cv2.imwrite(image_path, frame_copy)

                        # Save label in YOLO format
                        label_path = os.path.join(dirs["labels_dir"], f"{timestamp}.txt")
                        with log_lock, open(label_path, 'w') as f:
                            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# Consumer thread for YOLO models
def model_frame_consumer(model_name, model, queue, stop_event):
    print(f"Starting detection for {model_name} model...")
    while not stop_event.is_set() or not queue.empty():
        if not queue.empty():
            frame = queue.get()
            process_frame(
                frame,
                model,
                confidence_ranges_dirs[model_name],
                model.names
            )

# Main function
def main():
    queue = Queue(maxsize=10)
    stop_event = threading.Event()

    # Start webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    # Consumer threads for each model
    consumer_threads = []
    for model_name, model in models.items():
        thread = threading.Thread(target=model_frame_consumer, args=(model_name, model, queue, stop_event))
        consumer_threads.append(thread)
        thread.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break

            if not queue.full():
                queue.put(frame)

            cv2.imshow("Webcam Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping all threads...")
        stop_event.set()

    cap.release()
    cv2.destroyAllWindows()

    for thread in consumer_threads:
        thread.join()

    print("All threads stopped.")

if _name_ == "_main_":
    main()