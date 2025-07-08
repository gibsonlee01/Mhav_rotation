import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
import re
from pathlib import Path

def natural_sort_key(s):
    """
    자연 정렬을 위한 키를 반환하는 함수.
    예: "frame10.jpg"는 "frame2.jpg"보다 뒤에 오도록 정렬합니다.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]

def process_dataset_and_save_trajectories():
    """
    지정된 기본 폴더 내의 모든 액션에 대해 손 궤적을 추출하고 .npy 파일로 저장합니다.
    """
    # --- ⚙️ 사용자 설정 영역 ⚙️ ---
    base_input_folder = "/media/kcy/3A72CA7272CA3285/data_MHAV"
    track_hand = 'right'  # 추적할 손: 'right' 또는 'left'
    num_frames = 500      # 저장할 최종 프레임 수
    # ------------------------------------

    # 1. 모델 초기화
    print("🚀 WILOR 모델을 초기화합니다. 잠시만 기다려 주세요...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # WiLoR-mini는 float32를 사용합니다.
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32, verbose=False)
    print(f"✅ 모델 초기화 완료. ({device} 사용)")

    # 2. 처리할 액션 폴더 목록 탐색
    # 각 액션의 'RGB_undistorted' 폴더를 찾습니다.
    try:
        image_folders = sorted(Path(base_input_folder).glob('**/RGB_undistorted'))
    except FileNotFoundError:
        print(f"❌ 오류: 최상위 입력 폴더를 찾을 수 없습니다: {base_input_folder}")
        return

    if not image_folders:
        print(f"❌ 오류: '{base_input_folder}' 안에서 'RGB_undistorted' 폴더를 찾을 수 없습니다.")
        return

    print(f"\n총 {len(image_folders)}개의 액션 폴더를 처리합니다.")
    
    # 3. 각 액션 폴더별로 파이프라인 실행
    for image_folder in tqdm(image_folders, desc="전체 액션 진행률"):
        action_folder = image_folder.parent
        action_name = f"{action_folder.parent.name}_{action_folder.name}"
        tqdm.write("-" * 70)
        tqdm.write(f"Processing action: {action_name}")

        image_paths = sorted(image_folder.glob('*.png'), key=natural_sort_key)
        if not image_paths:
            image_paths = sorted(image_folder.glob('*.jpg'), key=natural_sort_key)
        
        if not image_paths:
            tqdm.write(f"⚠️ 경고: '{action_name}'에 이미지가 없어 건너뜁니다.")
            continue
        
        # 각 액션의 손목 좌표를 저장할 리스트
        hand_path_points = []
        for img_path in tqdm(image_paths, desc=f"Pose Estimation ({action_name})", leave=False):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # WiLoR로 손 예측 (cv2로 읽은 BGR 프레임 직접 사용)
            outputs = pipe.predict(image)
            
            target_hand_output = None
            if outputs:
                for out in outputs:
                    if (track_hand == 'right' and out['is_right'] == 1) or \
                       (track_hand == 'left' and out['is_right'] == 0):
                        target_hand_output = out
                        break
            
            if target_hand_output:
                wrist_keypoint_2d = target_hand_output["wilor_preds"]["pred_keypoints_2d"][0][0]
                center_x, center_y = int(wrist_keypoint_2d[0]), int(wrist_keypoint_2d[1])
                hand_path_points.append([center_x, center_y])

        # --- NPY 파일 저장 로직 ---
        if not hand_path_points:
            tqdm.write(f"⚠️ 경고: '{action_name}'에서 손이 전혀 감지되지 않아 .npy 파일을 생성하지 않습니다.")
            continue

        total_detected_frames = len(hand_path_points)
        points_to_process = np.array(hand_path_points, dtype=np.float32)
        final_path_array = None

        if total_detected_frames >= num_frames:
            # 중앙에서 num_frames 만큼 자르기
            start_index = (total_detected_frames - num_frames) // 2
            end_index = start_index + num_frames
            final_path_array = points_to_process[start_index:end_index]
        else:
            # 부족한 만큼 뒷부분을 마지막 좌표로 채우기
            padding_needed = num_frames - total_detected_frames
            last_coordinate = points_to_process[-1]
            back_padding = np.tile(last_coordinate, (padding_needed, 1))
            final_path_array = np.concatenate([points_to_process, back_padding], axis=0)
        
        # .npy 파일 저장 (액션 폴더 내에 저장)
        output_filename = f"{action_name}_wrist_trajectory.npy"
        output_path = action_folder / output_filename
        np.save(output_path, final_path_array)
        tqdm.write(f"✅ 궤적 추출 완료 및 저장: {output_path}")
        tqdm.write(f"   (감지된 프레임: {total_detected_frames}, 저장된 형태: {final_path_array.shape})")

    print("\n🎉 모든 작업이 완료되었습니다.")


if __name__ == '__main__':
    process_dataset_and_save_trajectories()