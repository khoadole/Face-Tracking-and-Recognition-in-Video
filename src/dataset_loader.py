## dataset_loader.py

import numpy as np

class DatasetLoader:
    def __init__(self, config: dict):
        self.batch_size = config['training']['batch_size']
        self.pose_clusters = config['training']['pose_clusters']
        self.pose_subspace_dimension = config['training']['pose_subspace_dimension']
        self.lmt_features_dimension = config['training']['lmt_features_dimension']
        self.pca_reduced_dimension = config['training']['pca_reduced_dimension']
    
    def load_data(self) -> dict:
        # Load face images from the datasets
        face_images = self.load_face_images()
        
        # Preprocess the face images
        preprocessed_data = self.preprocess_data(face_images)
        
        return preprocessed_data
    
    def load_face_images(self) -> np.array:
        # Load face images from the dataset
        # Include code to load face images
        face_images = np.random.rand(self.batch_size, 224, 224, 3)  # Dummy data for illustration
        return face_images
    
    def preprocess_data(self, face_images: np.array) -> dict:
        # Preprocess the face images (dummy preprocessing for illustration)
        preprocessed_data = {"images": face_images, "metadata": {"clusters": self.pose_clusters, 
                                                                 "subspace_dim": self.pose_subspace_dimension,
                                                                 "lmt_dim": self.lmt_features_dimension,
                                                                 "pca_dim": self.pca_reduced_dimension}}
        return preprocessed_data
