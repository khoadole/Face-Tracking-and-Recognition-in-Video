## main.py

import yaml
from dataset_loader import DatasetLoader
from model import PoseSubspaceModel, AlignmentConstraintModel, FaceRecognitionHMM
from face_tracker import FaceTracker
from face_recognizer import FaceRecognizer

# Load configuration settings from config.yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Step 1: Load the Dataset
data_loader = DatasetLoader(config)
preprocessed_data = data_loader.load_data()

# Step 2: Initialize Models
pose_subspace_model = PoseSubspaceModel(config['training']['pose_clusters'], config['training']['pose_subspace_dimension'])
alignment_model = AlignmentConstraintModel(config['training']['lmt_features_dimension'])
recognition_model = FaceRecognitionHMM(config['training']['pca_reduced_dimension'])

# Step 3: Face Tracking
face_tracker = FaceTracker({
    'pose_subspace_model': pose_subspace_model,
    'alignment_constraint_model': alignment_model,
    'lambda_a': config.get('tracking', {}).get('lambda_a', 1.0),
    'lambda_p': config.get('tracking', {}).get('lambda_p', 1.0),
    'lambda_s': config.get('tracking', {}).get('lambda_s', 1.0)
})

tracked_faces = []
for image in preprocessed_data['images']:
    result = face_tracker.track_face(image)
    tracked_faces.append(result)

# Step 4: Face Recognition
face_recognizer = FaceRecognizer(recognition_model)
recognized_identities = face_recognizer.recognize_faces(tracked_faces)

# Step 5: Display Results
for face_id, identity in recognized_identities.items():
    print(f"Face {face_id} identified as: {identity}")
