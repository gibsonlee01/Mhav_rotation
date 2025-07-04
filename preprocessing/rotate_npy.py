import os
import re
from pathlib import Path
import cv2
import torch
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as ScipyRotation
from tqdm import tqdm

# wilor_mini 라이브러리가 설치되어 있어야 합니다.
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

def natural_sort_key(s):
    """'frame1.jpg', 'frame2.jpg', 'frame10.jpg' 순으로 정렬합니다."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', str(s))]

def extract_rotation_feature_from_keypoints(
    keypoints_list: list,
    track_hand: str = 'right',
    num_frames: int = 500,
    smoothing_window_size: int = 5
):
    """
    메모리에 있는 키포인트 리스트로부터 회전각 특징 벡터를 추출합니다.
    """
    # 양손이 모두 검출된 프레임의 키포인트만 필터링
    valid_keypoints = []
    for kps in keypoints_list:
        right_hand_kps = kps[:21, :]
        left_hand_kps = kps[21:, :]
        is_right_detected = not np.all(right_hand_kps == 0)
        is_left_detected = not np.all(left_hand_kps == 0)

        if is_right_detected and is_left_detected:
            if track_hand == 'right':
                valid_keypoints.append(right_hand_kps)
            else:
                valid_keypoints.append(left_hand_kps)
    
    if not valid_keypoints:
        tqdm.write(f"Warning: 유효한 양손 키포인트를 찾지 못했습니다. 0 벡터를 반환합니다.")
        return np.zeros(num_frames)
        
    # 중앙에서 num_frames 만큼 자르기
    total_valid_frames = len(valid_keypoints)
    if total_valid_frames <= num_frames:
        keypoints_to_process = valid_keypoints
    else:
        start_index = (total_valid_frames - num_frames) // 2
        end_index = start_index + num_frames
        keypoints_to_process = valid_keypoints[start_index:end_index]

    # 회전각 계산
    last_grasp_points = None
    angle_history = []
    signed_accumulated_angle_deg = 0.0
    for keypoints_3d in keypoints_to_process:
        palm_center = (keypoints_3d[0] + keypoints_3d[5] + keypoints_3d[17]) / 3
        finger_center = (keypoints_3d[4] + keypoints_3d[8] + keypoints_3d[12]) / 3
        palm_to_finger_vector = finger_center - palm_center
        current_grasp_points = np.array([keypoints_3d[4], keypoints_3d[8], keypoints_3d[12]])
        if last_grasp_points is not None:
            delta_rotation, _ = ScipyRotation.align_vectors(current_grasp_points, last_grasp_points)
            rotvec = delta_rotation.as_rotvec()
            angle_increment_rad = np.linalg.norm(rotvec)
            sign = np.sign(np.dot(rotvec, palm_to_finger_vector)) if angle_increment_rad > 1e-6 else 0
            signed_delta_angle_deg = np.rad2deg(angle_increment_rad) * sign
            signed_accumulated_angle_deg += signed_delta_angle_deg
        angle_history.append(signed_accumulated_angle_deg)
        last_grasp_points = current_grasp_points.copy()

    # 패딩
    feature_vector = np.array(angle_history)
    if len(feature_vector) < num_frames:
        padding_value = feature_vector[-1] if len(feature_vector) > 0 else 0
        padding = np.full(num_frames - len(feature_vector), padding_value)
        feature_vector = np.concatenate([feature_vector, padding])

    # 평활화
    if smoothing_window_size > 1:
        feature_vector = pd.Series(feature_vector).rolling(
            window=smoothing_window_size, min_periods=1, center=True
        ).mean().to_numpy()

    return feature_vector

def process_dataset_and_save_features():
    # --- ⚙️ 사용자가 설정할 부분 ⚙️ ---
    base_input_folder = "/media/kcy/3A72CA7272CA3285/data_MHAV"
    track_hand = 'right'
    num_frames = 500
    smoothing_window = 5
    # ------------------------------------

    # 1. 모델 초기화
    print("🚀 WILOR 모델을 초기화합니다. 잠시만 기다려 주세요...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float16, verbose=False)
    print("✅ 모델 초기화 완료.")

    # 2. 처리할 액션 폴더 목록 탐색
    try:
        image_folders = sorted(Path(base_input_folder).glob('**/RGB_undistorted/processed_270_480'))
        action_folders = sorted(list(set([p.parent.parent for p in image_folders])))
    except FileNotFoundError:
        print(f"Error: 최상위 입력 폴더를 찾을 수 없습니다: {base_input_folder}")
        return

    if not action_folders:
        print(f"Error: '{base_input_folder}' 안에서 처리할 이미지 폴더를 찾을 수 없습니다.")
        return

    print(f"\n총 {len(action_folders)}개의 액션 폴더를 처리합니다.")
    
    # 3. 각 액션 폴더별로 파이프라인 실행
    for action_folder in tqdm(action_folders, desc="전체 액션 진행률"):
        action_name = f"{action_folder.parent.name}_{action_folder.name}"
        tqdm.write(f"\nProcessing action: {action_name}")

        image_folder = action_folder / "RGB_undistorted/processed_270_480"
        image_paths = sorted(image_folder.glob('*.jpg'), key=natural_sort_key)
        
        if not image_paths:
            tqdm.write(f"Warning: '{action_name}'에 이미지가 없어 건너뜁니다.")
            continue
        
        all_frame_keypoints = []
        for img_path in tqdm(image_paths, desc=f"Pose Estimation ({action_name})", leave=False):
            image = cv2.imread(str(img_path))
            if image is None: continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            outputs = pipe.predict(image_rgb)
            
            if not outputs:
                all_keypoints = np.zeros((42, 3), dtype=np.float32)
            else:
                missing_hand_placeholder = np.zeros((21, 3), dtype=np.float32)
                right_hand_kps, left_hand_kps = missing_hand_placeholder, missing_hand_placeholder
                for hand_data in outputs:
                    keypoints_full = hand_data['wilor_preds']['pred_keypoints_3d'][0]
                    keypoints = keypoints_full[:, :3]
                    if hand_data['is_right']: right_hand_kps = keypoints
                    else: left_hand_kps = keypoints
                all_keypoints = np.concatenate([right_hand_kps, left_hand_kps], axis=0)
            
            all_frame_keypoints.append(all_keypoints)
        
        rotation_feature = extract_rotation_feature_from_keypoints(
            keypoints_list=all_frame_keypoints,
            track_hand=track_hand,
            num_frames=num_frames,
            smoothing_window_size=smoothing_window
        )
        
        # ✅ .npy 파일 저장 위치를 각 액션 폴더 바로 아래로 변경
        output_filename = f"{action_name}.npy"
        output_path = action_folder / output_filename
        np.save(output_path, rotation_feature)
        tqdm.write(f"✅ 특징 추출 완료 및 저장: {output_path}")

    print("\n🎉 모든 작업이 완료되었습니다.")


if __name__ == '__main__':
    process_dataset_and_save_features()