## face_recognizer.py

import numpy as np
from model import FaceRecognitionHMM, extract_features

class FaceRecognizer:
    def __init__(self, model: FaceRecognitionHMM):
        self.model = model
    
    def recognize_faces(self, tracked_faces: list) -> dict:
        # Extract features for each tracked face
        face_features = [extract_features(face['image']) for face in tracked_faces]
        
        # Initialize dictionary for recognized identities
        recognized_identities = {}
        
        for idx, face_feature in enumerate(face_features):
            # Predict subject identity using HMM-based framework
            predicted_identity = self.model.forward_algorithm(face_feature)
            recognized_identities[f'Face_{idx+1}'] = predicted_identity
        
        return recognized_identities
