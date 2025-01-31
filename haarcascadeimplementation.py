import cv2
import numpy as np
from deepface import DeepFace
import os
import pickle
from scipy.spatial.distance import cosine

# Directory to save embeddings
EMBEDDINGS_DIR = "C:/face_embeddings4"  # Change to your desired directory

# Ensure embeddings directory exists
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# 1. Face Detection using Haar Cascade
class FaceDetector:
    def __init__(self, cascade_path="haarcascade_frontalface_default.xml"):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

    def detect_faces(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        detected_faces = []
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]
            detected_faces.append((face, (x, y, w, h)))
        return detected_faces

# 2. Generate Facial Embeddings using DeepFace
class FaceEmbedding:
    def __init__(self, model_name="Facenet"):
        self.model_name = model_name

    def generate_embedding(self, face_image):
        # DeepFace preprocess requires BGR -> RGB conversion
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        embeddings = DeepFace.represent(face_image_rgb, model_name=self.model_name, enforce_detection=False)
        embedding = embeddings[0]["embedding"]
        return embedding

# 3. Compare Faces and Return Statistics
class FaceComparator:
    def __init__(self, embeddings_dir):
        self.embeddings_dir = embeddings_dir
        self.embeddings = self.load_existing_embeddings()

    def load_existing_embeddings(self):
        embeddings = {}
        for file in os.listdir(self.embeddings_dir):
            path = os.path.join(self.embeddings_dir, file)
            with open(path, 'rb') as f:
                embedding = pickle.load(f)
                embeddings[file] = embedding
        return embeddings

    def save_embedding(self, name, embedding, image_name):
        # Save the embedding with the image name
        base_name = os.path.splitext(image_name)[0]  # Get base name of the image file
        path = os.path.join(self.embeddings_dir, f"{base_name}_{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(embedding, f)

    def compare(self, new_embedding, threshold=0.6):
        for name, embedding in self.embeddings.items():
            distance = cosine(np.array(embedding), np.array(new_embedding))
            if distance < threshold:  # Match found
                return name, distance
        return None, None

# Main Script
if __name__ == "__main__":
    # Initialize objects
    detector = FaceDetector()
    embedder = FaceEmbedding()
    comparator = FaceComparator(EMBEDDINGS_DIR)

    # Directory containing images to process
    images_dir = "C:/images"  # Replace with your images folder path
    output_dir = "C:/Users/Hp/Desktop/face-recognition/output111"  # Directory to save output images
    os.makedirs(output_dir, exist_ok=True)

    known_faces = {}
    unknown_faces = 0

    # Loop through all images in the directory
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)

        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Skip non-image files
            continue

        # Load the image
        image = cv2.imread(image_path)

        # Detect faces in the image
        faces = detector.detect_faces(image)

        for i, (face, (x, y, w, h)) in enumerate(faces):
            # Generate embedding for the detected face
            embedding = embedder.generate_embedding(face)

            # Compare embedding with existing known faces
            name, distance = comparator.compare(embedding)

            if name:
                # Known face
                known_faces[name] = known_faces.get(name, 0) + 1
                label = f"{name} ({distance:.2f})"
            else:
                # Unknown face
                unknown_faces += 1
                name = f"unknown_{unknown_faces}"
                comparator.save_embedding(name, embedding, image_file)
                label = "Unknown"

            # Draw the bounding box and label on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save the processed image
        output_path = os.path.join(output_dir, f"processed_{image_file}")
        cv2.imwrite(output_path, image)

        print(f"Processed {image_file} and saved to {output_path}")

    # Display final statistics
    print("\nFinal Statistics:")
    print("Known Faces:")
    for name, count in known_faces.items():
        print(f"  {name}: {count}")
    print(f"Unknown Faces: {unknown_faces}")
