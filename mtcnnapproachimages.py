import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
import os
import pickle
from scipy.spatial.distance import cosine

# Directory to save embeddings
EMBEDDINGS_DIR = "C:/face_embeddings2"  # Change to your desired directory

# Ensure embeddings directory exists
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# 1. Face Detection using MTCNN
class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, image):
        results = self.detector.detect_faces(image)
        faces = []
        for result in results:
            x, y, w, h = result['box']
            x, y = max(0, x), max(0, y)  # Handle negative values
            face = image[y:y + h, x:x + w]
            faces.append((face, (x, y, w, h)))
        return faces

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

    def save_embedding(self, name, embedding):
        path = os.path.join(self.embeddings_dir, f"{name}.pkl")
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

    # Load input image
    image_path = "C:/Users/Hp/Desktop/face-recognition/images.jpg"  # Replace with your image path
    image = cv2.imread(image_path)

    # Detect faces
    faces = detector.detect_faces(image)

    known_faces = {}
    unknown_faces = 0

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
            comparator.save_embedding(name, embedding)
            label = "Unknown"

        # Draw the bounding box and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display statistics about known and unknown faces
    print("Statistics:")
    print("Known Faces:")
    for name, count in known_faces.items():
        print(f"  {name}: {count}")
    print(f"Unknown Faces: {unknown_faces}")

    # Show the image with bounding boxes and labels
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
