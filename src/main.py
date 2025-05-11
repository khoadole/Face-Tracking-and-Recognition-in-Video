## main_fixed.py
import os
import numpy as np
import yaml
import cv2
import argparse
from dataset_loader import DatasetLoader
from face_tracker import FaceTracker
from model_ import AdaptiveAppearanceModel, PoseSubspaceModel, AlignmentConstraintModel

def main():
    parser = argparse.ArgumentParser(
        description='Face Tracking Implementation based on IVT with Visual Constraints')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--video', type=str, default='', help='Path to input video (optional)')
    parser.add_argument('--output', type=str, default='output', help='Path to output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize tracking results')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    os.makedirs(args.output, exist_ok=True)

    # Initialize face detector for robust initial state
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Loading dataset...")
    data_loader = DatasetLoader(config)
    print("Training face tracking models...")
    training_data = data_loader.load_data()

    pose_model = PoseSubspaceModel(
        n_poses=config['training']['pose_clusters'],
        n_components=config['training']['pose_subspace_dimension']
    )
    pose_model.train(training_data['pose_clusters'])

    alignment_model = AlignmentConstraintModel(
        feature_dim=config['training']['lmt_features_dimension']
    )
    alignment_model.train(
        training_data['aligned_faces'],
        training_data['misaligned_faces']
    )

    adaptive_model = AdaptiveAppearanceModel(n_components=config['tracking']['adaptive_components'])

    tracker_params = {
        'pose_subspace_model': pose_model,
        'alignment_constraint_model': alignment_model,
        'adaptive_appearance_model': adaptive_model,
        'lambda_a': config['tracking']['lambda_a'],
        'lambda_p': config['tracking']['lambda_p'],
        'lambda_s': config['tracking']['lambda_s'],
        'n_particles': config['tracking']['n_particles'],
        'sigma': config['tracking']['sigma'],
        'std_size': tuple(config['dataset']['std_face_size'])
    }
    face_tracker = FaceTracker(tracker_params)

    if args.video:
        print(f"Tracking faces in video: {args.video}")
        track_video(face_tracker, args.video, args.output, args.visualize, face_cascade)
    else:
        print("Tracking faces in all videos in the dataset...")
        video_paths = data_loader.get_video_paths()
        if not video_paths:
            print("No videos found")
        else:
            for video_path in video_paths:
                print(f"Processing video: {video_path}")
                track_video(face_tracker, video_path, args.output, args.visualize, face_cascade)

    print("Face tracking completed.")

def track_video(face_tracker, video_path, output_dir, visualize=False, face_cascade=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_name = os.path.splitext(os.path.basename(video_path))[0]
    # out = cv2.VideoWriter(os.path.join(output_dir, f"{video_name}_tracked.mp4"), fourcc, fps, (width, height))
    folder_name = os.path.basename(os.path.dirname(video_path))
    out = cv2.VideoWriter(os.path.join(output_dir, f"{folder_name}_tracked.mp4"), fourcc, fps, (width, height))

    # Read and detect in the first frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Cannot read first frame from {video_path}")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        c_x = x + w/2; c_y = y + h/2
        # rho = max(w, h) / max(width, height)
        rho = max(w, h) / min(width, height)
        phi = 0.0
        print(f"Initial face at {x},{y},{w},{h} -> state: {[c_x, c_y, rho, phi]}")
    else:
        c_x, c_y = width/2, height/2
        rho, phi = 0.2, 0.0
        print("No face detected; using center as initial state.")

    face_tracker.initialize(np.array([c_x, c_y, rho, phi]))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = face_tracker.track_face(frame)
        state = result['tracking_state']
        # print(f"[DEBUG] state: c_x={state['c_x']:.1f}, c_y={state['c_y']:.1f}, rho={state['rho']:.3f}, phi={state['phi']:.3f}")
        cx, cy = int(state['c_x']), int(state['c_y'])
        rho, phi = state['rho'], state['phi']

        if visualize:
            box_size = int(min(width, height) * rho)
            half = box_size // 2
            pts = np.array([[cx-half, cy-half], [cx+half, cy-half], [cx+half, cy+half], [cx-half, cy+half]], np.int32)
            rot = cv2.getRotationMatrix2D((cx, cy), np.degrees(phi), 1.0)
            pts = pts.reshape(-1,1,2)
            pts = cv2.transform(pts, rot)
            cv2.polylines(frame, [pts], True, (0,255,0), 2)
            cv2.putText(frame, f"E={result['energy']:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cf = result['cropped_face']; h0,w0 = cf.shape[:2]
            # print(f"[DEBUG] cropped_face shape: {cf.shape}")
            frame[10:10+h0, width-10-w0:width-10] = cf

        out.write(frame)
        frame_idx += 1

    cap.release(); out.release()
    print(f"Finished {folder_name}, frames: {frame_idx}")
    return

if __name__ == '__main__':
    main()