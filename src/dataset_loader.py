## dataset_loader.py

import numpy as np
import os
import cv2
import glob
from typing import List, Dict, Tuple, Any

class DatasetLoader:
    def __init__(self, config: dict):
        # Configuration parameters
        self.batch_size = config['training']['batch_size']
        self.pose_clusters = config['training']['pose_clusters']
        self.pose_subspace_dimension = config['training']['pose_subspace_dimension']
        self.lmt_features_dimension = config['training']['lmt_features_dimension']
        self.pca_reduced_dimension = config['training']['pca_reduced_dimension']
        
        # Data paths
        self.dataset_root = config.get('dataset', {}).get('root_path', 'data')
        self.dataset_name = config.get('dataset', {}).get('name', '300VW_Dataset_2015_12_14')
        
        # Cache for loaded data
        self.cached_data = None
        
        # Initial face size
        self.standard_face_size = config.get('dataset', {}).get('std_face_size', (48, 48))
    
    def load_data(self) -> dict:
        """Load and preprocess face data from the 300VW dataset"""
        if self.cached_data is not None:
            return self.cached_data
        
        # Load facial landmark points from the dataset
        landmarks_data = self.load_landmarks()
        
        # Load corresponding images if available
        images_data = self.load_images(landmarks_data)
        
        # Split data by poses for pose subspace model training
        pose_clusters = self.cluster_by_pose(images_data, landmarks_data)
        
        # Generate misaligned samples for alignment constraint model training
        aligned_faces, misaligned_faces = self.generate_alignment_samples(images_data)
        
        # Create the preprocessed dataset
        preprocessed_data = {
            "images": images_data,
            "landmarks": landmarks_data,
            "pose_clusters": pose_clusters,
            "aligned_faces": aligned_faces,
            "misaligned_faces": misaligned_faces,
            "metadata": {
                "clusters": self.pose_clusters,
                "subspace_dim": self.pose_subspace_dimension,
                "lmt_dim": self.lmt_features_dimension,
                "pca_dim": self.pca_reduced_dimension
            }
        }
        
        # Cache the data
        self.cached_data = preprocessed_data
        
        return preprocessed_data
    
    def load_landmarks(self) -> List[np.ndarray]:
        """Load facial landmark points from .pts files in the 300VW dataset"""
        landmarks_data = []
        
        # Get all .pts files in the dataset
        dataset_path = os.path.join(self.dataset_root, self.dataset_name)
        pts_files = []
        
        # Recursively find all .pts files
        for dirpath, _, filenames in os.walk(dataset_path):
            for f in filenames:
                if f.endswith('.pts'):
                    pts_files.append(os.path.join(dirpath, f))
        
        # Limit to batch size if needed
        if self.batch_size > 0 and len(pts_files) > self.batch_size:
            pts_files = pts_files[:self.batch_size]
        
        # Load each landmark file
        for pts_file in pts_files:
            landmarks = self.read_pts_file(pts_file)
            if landmarks is not None:
                landmarks_data.append(landmarks)
        
        return landmarks_data
    
    def read_pts_file(self, pts_file: str) -> np.ndarray:
        """Read landmark points from a .pts file in the 300VW dataset format"""
        try:
            with open(pts_file, 'r') as f:
                lines = f.readlines()
            
            # Extract the points
            points = []
            n_points = 0
            reading_points = False
            
            for line in lines:
                line = line.strip()
                
                # Check if we found the version line
                if line.startswith('version:'):
                    continue
                    
                # Check if we found the n_points line
                elif line.startswith('n_points:'):
                    n_points = int(line.split(':')[1].strip())
                    continue
                    
                # Check for the start of points data
                elif line == '{':
                    reading_points = True
                    continue
                    
                # Check for the end of points data
                elif line == '}':
                    reading_points = False
                    continue
                
                # Read points data
                if reading_points and line:
                    try:
                        x, y = map(float, line.split())
                        points.append([x, y])
                    except ValueError:
                        continue
            
            # Verify that we read the correct number of points
            if len(points) == n_points:
                return np.array(points)
            else:
                print(f"Warning: Expected {n_points} points but read {len(points)} from {pts_file}")
                return None
                
        except Exception as e:
            print(f"Error reading {pts_file}: {e}")
            return None
    
    def load_images(self, landmarks_data: List[np.ndarray]) -> List[np.ndarray]:
        """Load corresponding images for the landmarks data"""
        images_data = []
        
        # Since we don't have direct access to the images in the dataset,
        # we'll create dummy images with the landmarks plotted for demonstration
        for landmarks in landmarks_data:
            # Create a blank image
            img_size = (max(int(np.max(landmarks[:, 0])) + 50, 640), 
                       max(int(np.max(landmarks[:, 1])) + 50, 480))
            img = np.ones((*img_size, 3), dtype=np.uint8) * 255
            
            # Draw landmarks
            for i, (x, y) in enumerate(landmarks):
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            
            # Extract face region using landmarks
            if len(landmarks) > 0:
                x_min, y_min = np.min(landmarks, axis=0)
                x_max, y_max = np.max(landmarks, axis=0)
                
                # Add margin
                margin = 30
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(img.shape[1] - 1, x_max + margin)
                y_max = min(img.shape[0] - 1, y_max + margin)
                
                # Crop face region
                face_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
                
                # Resize to standard size
                face_img = cv2.resize(face_img, self.standard_face_size)
                
                images_data.append(face_img)
        
        return images_data
    
    def cluster_by_pose(self, images: List[np.ndarray], landmarks: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Cluster face images by pose using landmarks
        
        Returns:
            List of lists, where each inner list contains images of a specific pose cluster
        """
        pose_clusters = [[] for _ in range(self.pose_clusters)]
        
        # Simplified pose estimation based on landmark patterns
        # In a real implementation, use more sophisticated methods
        for i, (image, lm) in enumerate(zip(images, landmarks)):
            if len(lm) == 0:
                continue
                
            # Determine pose based on landmarks distribution
            # (This is simplified - in real implementation use more accurate pose estimation)
            # Calculate center of landmarks
            center_x = np.mean(lm[:, 0])
            
            # Calculate the distribution of landmarks relative to center
            x_distribution = lm[:, 0] - center_x
            left_weight = np.sum(x_distribution < 0)
            right_weight = np.sum(x_distribution > 0)
            
            # Simple pose estimation based on landmark distribution
            if abs(left_weight - right_weight) < len(lm) * 0.2:
                # Roughly balanced - assign to frontal pose (cluster 0)
                pose_clusters[0].append(image)
            elif left_weight > right_weight * 2:
                # More landmarks on left - likely right-facing (cluster 1)
                pose_clusters[1].append(image)
            elif right_weight > left_weight * 2:
                # More landmarks on right - likely left-facing (cluster 2)
                pose_clusters[2].append(image)
            elif left_weight > right_weight * 1.5:
                # Slightly more on left - right 45-deg (cluster 3)
                pose_clusters[3].append(image)
            elif right_weight > left_weight * 1.5:
                # Slightly more on right - left 45-deg (cluster 4)
                pose_clusters[4].append(image)
            else:
                # Default to frontal
                pose_clusters[0].append(image)
        
        return pose_clusters
    
    def generate_alignment_samples(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate well-aligned and misaligned face samples for training the alignment constraint model
        
        Returns:
            Tuple containing:
                - List of well-aligned face images
                - List of misaligned face images
        """
        aligned_faces = []
        misaligned_faces = []
        
        for image in images:
            if image is None or image.size == 0:
                continue
                
            # Add original image as well-aligned
            aligned_faces.append(image.copy())
            
            # Generate misaligned versions by random transformations
            for _ in range(2):  # Generate 2 misaligned versions per image
                # Random shift, scale, and rotation
                rows, cols = image.shape[:2]
                
                # Random shift
                tx = np.random.randint(-cols//6, cols//6)
                ty = np.random.randint(-rows//6, rows//6)
                
                # Random scale (0.7 to 1.3)
                scale = 0.7 + np.random.random() * 0.6
                
                # Random rotation (-45 to 45 degrees)
                angle = np.random.randint(-45, 45)
                
                # Create transformation matrix
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
                M[0, 2] += tx
                M[1, 2] += ty
                
                # Apply transformation
                misaligned_img = cv2.warpAffine(image, M, (cols, rows))
                
                misaligned_faces.append(misaligned_img)
        
        return aligned_faces, misaligned_faces
    
    # def load_video_frames(self, video_path: str, max_frames: int = 0) -> List[np.ndarray]:
    #     """Load frames from a video file for tracking
        
    #     Args:
    #         video_path: Path to the video file
    #         max_frames: Maximum number of frames to load (0 for all)
            
    #     Returns:
    #         List of video frames as numpy arrays
    #     """
    #     frames = []
        
    #     # Open the video file
    #     try:
    #         cap = cv2.VideoCapture(video_path)
    #         if not cap.isOpened():
    #             print(f"Error opening video file: {video_path}")
    #             return []
                
    #         frame_count = 0
    #         while True:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
                    
    #             frames.append(frame)
    #             frame_count += 1
                
    #             if max_frames > 0 and frame_count >= max_frames:
    #                 break
                    
    #         cap.release()
            
    #     except Exception as e:
    #         print(f"Error processing video {video_path}: {e}")
            
    #     return frames
    
    def get_video_paths(self) -> List[str]:
        """Get paths to all video files in the dataset
        
        Returns:
            List of paths to video files
        """
        video_paths = []
        
        # Get all video files in the dataset
        dataset_path = os.path.join(self.dataset_root, self.dataset_name)
        
        # Common video extensions
        video_extensions = ['*.avi', '*.mp4', '*.mov', '*.mkv']
        
        # Find all video files
        for ext in video_extensions:
            pattern = os.path.join(dataset_path, '**', ext)
            video_paths.extend(glob.glob(pattern, recursive=True))
        
        return video_paths