import pickle

# Path to the saved embedding file
embedding_path = "C:/face_embeddings2/unknown_1.pkl"  # Example path

# Load the saved embedding
with open(embedding_path, 'rb') as f:
    embedding = pickle.load(f)

# Print the embedding (numerical array)
print(embedding)
