import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as ScipyRotation
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Helper Function for Natural Sorting ---
def natural_sort_key(s):
    """
    Sorts strings containing numbers in a natural way.
    E.g., "frame1.csv", "frame2.csv", "frame10.csv"
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]

# --- Main Feature Extraction Function ---

def extract_rotation_feature_from_csv(
    keypoint_folder: str,
    video_name: str,
    track_hand: str = 'right',
    num_frames: int = 500,
    smoothing_window_size: int = 5 # ✅ 시간적 평활화를 위한 윈도우 크기
):
    """
    미리 계산된 3D 키포인트 CSV에서 회전각 특징 벡터를 추출합니다.
    - 양손이 모두 검출된 프레임만 사용합니다.
    - 최종 특징 벡터에 시간적 평활화를 적용합니다.

    Args:
        keypoint_folder (str): 'keypoint_*.csv' 파일들이 저장된 폴더 경로.
        video_name (str): 진행률 표시줄에 사용할 비디오 이름.
        track_hand (str, optional): 추적할 손. 'right' 또는 'left'. Defaults to 'right'.
        num_frames (int, optional): 고정할 특징 벡터의 길이. Defaults to 500.
        smoothing_window_size (int, optional): 이동 평균 필터의 윈도우 크기. 1 이하면 비활성화. Defaults to 5.

    Returns:
        np.ndarray: (num_frames,) 형태의 1D 회전각 특징 벡터.
    """
    csv_files = sorted(glob.glob(os.path.join(keypoint_folder, 'keypoint_*.csv')), key=natural_sort_key)
    if not csv_files:
        raise FileNotFoundError(f"키포인트 CSV 파일을 찾을 수 없음: '{keypoint_folder}'")
    
    total_files = len(csv_files)
    if total_files <= num_frames:
        # 전체 파일 수가 요청 프레임 수보다 적거나 같으면 전체 사용
        files_to_process = csv_files
    else:
        # 중앙에서 자르기 위한 시작점과 끝점 계산
        start_index = (total_files - num_frames) // 2
        end_index = start_index + num_frames
        files_to_process = csv_files[start_index:end_index]
            
    all_keypoints = []
    for csv_path in tqdm(files_to_process, desc=f"Extracting features from CSVs ({video_name})", leave=False):
        try:
            df = pd.read_csv(csv_path)
            all_kps_from_csv = df.to_numpy()

            right_hand_kps = all_kps_from_csv[:21, :]
            left_hand_kps = all_kps_from_csv[21:, :]

            # ✅ 1. 양손이 모두 검출되었는지 확인하는 로직
            is_right_detected = not np.all(right_hand_kps == 0)
            is_left_detected = not np.all(left_hand_kps == 0)

            # 양손이 모두 검출된 경우에만 키포인트를 추가합니다.
            if is_right_detected and is_left_detected:
                if track_hand == 'right':
                    all_keypoints.append(right_hand_kps)
                else: # 'left'
                    all_keypoints.append(left_hand_kps)

        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue

    if not all_keypoints:
        print(f"Warning: 양손이 모두 검출된 유효 프레임을 찾지 못했습니다 ({video_name}). 0 벡터를 반환합니다.")
        return np.zeros(num_frames)

    # ... 회전각 계산 로직 (이전과 동일) ...
    last_grasp_points = None
    angle_history = []
    signed_accumulated_angle_deg = 0.0
    for keypoints_3d in all_keypoints:
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

    # ... 패딩 로직 (이전과 동일) ...
    feature_vector = np.array(angle_history)
    if len(feature_vector) < num_frames:
        padding_value = feature_vector[-1] if len(feature_vector) > 0 else 0
        padding = np.full(num_frames - len(feature_vector), padding_value)
        feature_vector = np.concatenate([feature_vector, padding])
    
    # ✅ 2. 시간적 평활화 (Temporal Smoothing) 적용
    if smoothing_window_size > 1:
        # pandas의 rolling을 사용하면 간편하게 이동 평균을 계산할 수 있습니다.
        feature_vector = pd.Series(feature_vector).rolling(
            window=smoothing_window_size,
            min_periods=1,  # 윈도우가 다 채워지지 않아도 계산
            center=True     # 중앙 정렬로 필터링 결과의 시간 지연 방지
        ).mean().to_numpy()

    return feature_vector

# --- Example Usage ---
if __name__ == '__main__':
    base_data_folder = "/media/kcy/3A72CA7272CA3285/data_MHAV"
    subject_action_folder = "S01/A01_K_1"
    keypoint_folder_path = os.path.join(
        base_data_folder,
        subject_action_folder,
        "wilor_pose_3d/processed_270_480/keypoint"
    )

    print(f"Processing folder: {keypoint_folder_path}")

    try:
        # 평활화가 적용된 특징 벡터 추출
        smoothed_feature = extract_rotation_feature_from_csv(
            keypoint_folder=keypoint_folder_path,
            video_name=subject_action_folder,
            track_hand='right',
            num_frames=500,
            smoothing_window_size=11 # 홀수로 설정하는 것이 일반적
        )
        
        # 비교를 위해 평활화가 없는 원본 특징 벡터도 추출
        raw_feature = extract_rotation_feature_from_csv(
            keypoint_folder=keypoint_folder_path,
            video_name=subject_action_folder,
            track_hand='right',
            num_frames=500,
            smoothing_window_size=1 # 평활화 비활성화
        )

        print("\n✅ 특징 추출 완료!")
        print(f"Smoothed feature vector shape: {smoothed_feature.shape}")

        # 결과 시각화
        plt.figure(figsize=(14, 7))
        plt.plot(raw_feature, label='Original Feature', color='skyblue', alpha=0.8)
        plt.plot(smoothed_feature, label='Smoothed Feature (window=11)', color='coral', linewidth=2)
        plt.title(f"Rotation Angle Feature (Original vs. Smoothed) - {subject_action_folder}")
        plt.xlabel("Frame Index")
        plt.ylabel("Accumulated Angle (degrees)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        output_viz_path = os.path.join(os.path.dirname(keypoint_folder_path), "rotation_feature_smoothing_comparison.png")
        plt.savefig(output_viz_path)
        plt.close()
        print(f"\n📈 시각화 결과가 다음 경로에 저장되었습니다: {output_viz_path}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")