import numpy as np
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
import cv2




class AppearanceAdaptiveModel:
    def __init__(self, n_components=10, forgetting_factor=0.95):
        """
        Khởi tạo mô hình thích nghi appearance dùng SVD.
        
        Args:
            n_components (int): Số thành phần PCA.
            forgetting_factor (float): Hệ số quên (0 < ff <= 1), càng nhỏ càng nhanh quên dữ liệu cũ.
        """
        self.n_components = n_components
        self.forgetting_factor = forgetting_factor
        self.mean = None          # Mean vector (1D)
        self.basis = None         # PCA basis (n_components x d)
        self.singular_values = None  # Singular values

    def init(self, first_image):
        """Khởi tạo subspace từ ảnh đầu tiên."""
        flattened = first_image.flatten().astype(np.float32)
        self.mean = flattened
        self.basis = np.zeros((self.n_components, len(flattened)))
        self.singular_values = np.zeros(self.n_components)

    def update(self, new_image):
        """
        Cập nhật subspace với ảnh mới dùng incremental SVD.
        
        Args:
            new_image: Ảnh mới (grayscale, đã resize).
        """
        flattened = new_image.flatten().astype(np.float32)
        
        if self.mean is None:
            self.init(new_image)
            return

        # Chuẩn hóa ảnh mới bằng mean hiện tại
        centered = flattened - self.mean

        # Cập nhật mean với hệ số quên
        self.mean = self.forgetting_factor * self.mean + (1 - self.forgetting_factor) * flattened

        # Tính residual sau khi cập nhật mean
        residual = flattened - self.mean

        # Cập nhật basis bằng incremental SVD
        if self.basis is not None:
            # Tạo ma trận tạm chứa basis và residual
            temp = np.vstack([
                self.singular_values.reshape(-1, 1) * self.basis,
                residual[np.newaxis, :]
            ])
            
            # Tính SVD của ma trận tạm
            U, S, Vt = svd(temp, full_matrices=False)
            
            # Giữ lại n_components lớn nhất
            U = U[:, :self.n_components]
            S = S[:self.n_components]
            Vt = Vt[:self.n_components, :]
            
            # Cập nhật basis và singular values
            self.basis = Vt
            self.singular_values = S

    def reconstruct(self, image):
        """
        Tính reconstruction error của ảnh so với subspace hiện tại.
        
        Args:
            image: Ảnh đầu vào (grayscale, đã resize).
        Returns:
            error: Reconstruction error (càng nhỏ càng tốt).
        """
        flattened = image.flatten().astype(np.float32)
        centered = flattened - self.mean
        projection = self.basis @ (self.basis.T @ centered)
        return np.linalg.norm(centered - projection)
    


class PoseSubspaceModel:
    def __init__(self, pose_clusters: int, pose_subspace_dim: int):
        """
        Khởi tạo mô hình.        
        Args:
            pose_clusters (int): Số tư thế
            pose_subspace_dim (int): Số thành phần chính cho PCA.
        """
        self.pose_clusters = pose_clusters
        self.pose_subspace_dim = pose_subspace_dim
        # Initialize model parameters and optimizer if training is needed
        # Initialize model 
        self.pose_subspaces = []  # List chứa các tuple (mean, basis) cho mỗi tư thế
        self.kmeans = None



    def cluster_poses(self, images):
        """
        Phân cụm ảnh vào các tư thế sử dụng K-Means.
        
        Args:
            images (ndarray): Ma trận 2D (n_samples, n_pixels).
        """
        self.kmeans = KMeans(n_clusters=self.pose_clusters)
        self.kmeans.fit(images)


    def train_subspaces(self, images):
        """
        Huấn luyện PCA cho từng cụm tư thế.
        
        Args:
            images (ndarray): Ma trận 2D (n_samples, n_pixels), mỗi hàng là 1 frame trong video
        """
        labels = self.kmeans.labels_
        for pose_id in range(self.pose_clusters):
            pose_images = images[labels == pose_id]
            pca = PCA(n_components=self.pose_subspace_dim)
            pca.fit(pose_images)
            self.pose_subspaces.append((pca.mean_, pca.components_))


    def fit(self, images):
        """Pipeline huấn luyện đầy đủ."""
        self.cluster_poses(images)
        self.train_subspaces(images)



    def compute_distance(self, image, pose_id):
        """
        Tính khoảng cách từ ảnh đến không gian con PCA của tư thế `pose_id`.
        
        Args:
            image (ndarray): Ảnh đầu vào (đã làm phẳng) cần dự đoán
            pose_id (int): ID tư thế.
        Returns:
            float: Khoảng cách reconstruction error.
        """
        mean, basis = self.pose_subspaces[pose_id]
        centered = image - mean
        projection = basis @ (basis.T @ centered)
        return np.linalg.norm(centered - projection)
            
    
    def predict_distance(self, image: np.array) -> float:
        """
        Dự đoán tư thế và trả về khoảng cách tối thiểu.
        
        Args:
            image (ndarray): Ảnh đầu vào (48x48, grayscale).
        Returns:
            tuple: (pose_id, min_distance).
        """
        flattened = image.flatten()
        distances = [self.compute_distance(flattened, pose_id) for pose_id in range(self.n_poses)]
        return np.argmin(distances), min(distances)
    


class AlignmentConstraintModel:
    def __init__(self, lmt_features_dim: int):
        self.lmt_features_dim = lmt_features_dim
        # Initialize SVM model and parameters
        # Implement SVM setup and training process
        self.svm = SVC(kernel='rbf', probability=True)  # SVM với kernel RBF, hỗ trợ xác suất


    def generate_negative_samples(self, positive_images, n_negatives=1000):
        """
        Tạo ảnh negative (căn chỉnh kém) từ ảnh positive bằng cách áp dụng biến đổi ngẫu nhiên.
        
        Args:
            positive_images: List ảnh positive (đã flatten).
            n_negatives: Số ảnh negative cần tạo.
        Returns:
            negative_samples: Các ảnh negative (kích thước 48x48).
        """
        negative_samples = []
        for _ in range(n_negatives):
            # Chọn ngẫu nhiên một ảnh positive
            img = positive_images[np.random.randint(0, len(positive_images))].reshape(48, 48)
            
            # Áp dụng biến đổi affine ngẫu nhiên (dịch chuyển + xoay)
            dx, dy = np.random.randint(-15, 15, 2)  # Dịch ngẫu nhiên [-15, 15] pixels
            angle = np.random.randint(-30, 30)      # Xoay ngẫu nhiên [-30°, 30°]
            
            M = cv2.getRotationMatrix2D((24, 24), angle, 1.0)  # Tâm xoay là trung tâm ảnh
            M[:, 2] += [dx, dy]  # Thêm dịch chuyển
            distorted = cv2.warpAffine(img, M, (48, 48))
            
            negative_samples.append(distorted.flatten())
        
        return np.array(negative_samples)
    

    def prepare_data(self, positive_images: np.ndarray):
        """
        Chuẩn bị dữ liệu huấn luyện SVM (positive + negative).
        
        Args:
            positive_images: List ảnh positive (đã flatten).
        Returns:
            X_train, X_test, y_train, y_test: Dữ liệu đã chia train/test.
        """
        negatives = self.generate_negative_samples(positive_images)
        X = np.vstack([positive_images, negatives])
        y = np.array([1] * len(positive_images) + [0] * len(negatives))  # 1 = good, 0 = bad
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    

    
    def fit(self, positive_images: np.ndarray):
        """
        Huấn luyện SVM từ thư mục chứa ảnh positive (đã căn chỉnh).
        
        Args:
            positive_images: List ảnh positive (đã flatten).
        """
        
        X_train, X_test, y_train, y_test = self.prepare_data(positive_images)
        
        self.svm.fit(X_train, y_train)
        print(f"SVM Accuracy: {self.svm.score(X_test, y_test):.2f}")
        
    
    def calculate_confidence(self, face_image: np.array) -> float:
        """
        Đánh giá chất lượng căn chỉnh của ảnh đầu vào.
        
        Args:
            image: Ảnh khuôn mặt (48x48, grayscale).
        Returns:
            proba: Xác suất ảnh thuộc lớp "căn chỉnh tốt" [0, 1].
        """
        proba = self.svm.predict_proba(face_image.flatten().reshape(1, -1))
        return proba[0][1]  # P(class=1|image)

class FaceRecognitionHMM:
    def __init__(self, pca_reduced_dim: int):
        self.pca_reduced_dim = pca_reduced_dim
        # Setup HMM parameters and structures

    def initialize_hmm(self):
        # Initialize HMM parameters based on configuration settings
        pass

    def train_em(self, face_sequences: np.array):
        # Train the HMM using the EM algorithm and face sequence data
        # Implement EM training
        pass

    def forward_algorithm(self, face_sequence: np.array) -> np.array:
        # Implement the forward algorithm for state estimation and prediction
        # Return predicted classes for the face sequence
        pass

def extract_features(face_images: np.array, lmt_features_dim: int) -> np.array:
    # Extract features like LDA and LMT from face images
    # Apply PCA to reduce dimensionality if needed
    # Return the feature vectors
    pass

# In case additional functions or classes are needed based on design, add them below
