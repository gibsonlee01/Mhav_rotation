import os
import re
from pathlib import Path
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque

# wilor_mini 라이브러리가 설치되어 있어야 합니다.
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

def natural_sort_key(s):
    """'frame1.jpg', 'frame2.jpg', 'frame10.jpg' 순으로 정렬합니다."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', str(s))]

# ✅ --- 손목 회전 방향 특징만 추출하도록 함수 수정 ---
def extract_wrist_turn_feature_from_keypoints(
    keypoints_list: list,
    track_hand: str = 'right',
    num_frames: int = 500,
    smoothing_window_size: int = 5
):
    """
    메모리에 있는 키포인트 리스트로부터 손목 회전 방향 특징만 추출합니다.

    Returns:
        np.ndarray: (num_frames,) 형태의 1D 특징 벡터.
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

    # 손목 회전 방향 계산
    wrist_turn_history = []
    accumulated_turn_direction = 0.0
    wrist_pos_deque = deque(maxlen=3)

    for keypoints_3d in keypoints_to_process:
        current_wrist_pos = keypoints_3d[0]
        wrist_pos_deque.append(current_wrist_pos[:2]) # XY 평면에서의 움직임
        
        turn_direction = 0.0
        if len(wrist_pos_deque) == 3:
            p_prev, p_mid, p_curr = wrist_pos_deque
            v1, v2 = p_mid - p_prev, p_curr - p_mid
            # 2D 외적을 통해 회전 방향 판단
            cross_product_z = v1[0] * v2[1] - v1[1] * v2[0]
            turn_direction = np.sign(cross_product_z)
            
        accumulated_turn_direction += turn_direction
        wrist_turn_history.append(accumulated_turn_direction)

    # 1D 벡터로 변환
    feature_vector = np.array(wrist_turn_history)

    # 패딩
    if len(feature_vector) < num_frames:
        padding_value = feature_vector[-1] if len(feature_vector) > 0 else 0
        padding = np.full(num_frames - len(feature_vector), padding_value)
        feature_vector = np.concatenate([feature_vector, padding])

    # 평활화
    if smoothing_window_size > 1 and len(feature_vector) > 0:
        feature_vector = pd.Series(feature_vector).rolling(
            window=smoothing_window_size, min_periods=1, center=True).mean().to_numpy()

    return feature_vector

def process_dataset_and_save_features():
    # --- ⚙️ 사용자가 설정할 부분 ⚙️ ---
    base_input_folder = "/media/kcy/3A72CA7272CA3285/data_MHAV"
    track_hand = 'right'
    num_frames = 500
    smoothing_window = 5
    # ------------------------------------

    print("🚀 WILOR 모델을 초기화합니다. 잠시만 기다려 주세요...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float16, verbose=False)
    print("✅ 모델 초기화 완료.")

    try:
        image_folders = sorted(Path(base_input_folder).glob('**/RGB_undistorted/processed_270_480'))
        action_folders = sorted(list(set([p.parent.parent for p in image_folders])))
        
        action_folders = [folder for folder in action_folders if 'unscrew' in folder.name]
    except FileNotFoundError:
        print(f"Error: 최상위 입력 폴더를 찾을 수 없습니다: {base_input_folder}")
        return
    if not action_folders:
        print(f"Error: '{base_input_folder}' 안에서 처리할 이미지 폴더를 찾을 수 없습니다.")
        return

    print(f"\n총 {len(action_folders)}개의 액션 폴더를 처리합니다.")

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
        
        # ✅ --- 손목 회전 방향 특징 추출 함수 호출 ---
        wrist_turn_feature = extract_wrist_turn_feature_from_keypoints(
            keypoints_list=all_frame_keypoints,
            track_hand=track_hand,
            num_frames=num_frames,
            smoothing_window_size=smoothing_window
        )
        
        # ✅ .npy 파일 이름 수정
        output_filename = f"{action_name}_wrist_turn.npy"
        output_path = action_folder / output_filename
        np.save(output_path, wrist_turn_feature)
        tqdm.write(f"✅ 손목 회전 방향 특징 추출 완료 및 저장: {output_path}")

    print("\n🎉 모든 작업이 완료되었습니다.")


if __name__ == '__main__':
    process_dataset_and_save_features()