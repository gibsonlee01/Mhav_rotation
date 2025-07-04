import torch
import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from tqdm import tqdm # tqdm 라이브러리 import

# 화면에 그림을 표시하지 않고 파일로 바로 저장하기 위한 설정
matplotlib.use('Agg')

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

# 3D 스켈레톤을 그리기 위한 관절 연결 정보
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def process_and_visualize_all_folders():
    # --- ⚙️ 사용자가 설정할 부분 ⚙️ ---
    base_input_folder = "/media/kcy/3A72CA7272CA3285/data_MHAV"
    # ------------------------------------

    print("모델을 초기화합니다. 잠시만 기다려 주세요...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    print("모델 초기화 완료.")

    try:
        image_folders = sorted(Path(base_input_folder).glob('**/RGB_undistorted/processed_270_480'))
    except FileNotFoundError:
        print(f"Error: 최상위 입력 폴더를 찾을 수 없습니다: {base_input_folder}")
        return

    if not image_folders:
        print(f"Error: '{base_input_folder}' 안에서 '**/RGB_undistorted/processed_270_480' 폴더를 찾을 수 없습니다.")
        return

    print(f"\n총 {len(image_folders)}개의 이미지 폴더를 처리합니다.")
    
    for image_folder in tqdm(image_folders, desc="전체 폴더 진행률"):
        
        supported_extensions = ['*.jpg', '*.png', '*.jpeg']
        image_paths = sorted([p for ext in supported_extensions for p in image_folder.glob(ext)])
        if not image_paths:
            continue
            
        output_base_dir_root = image_folder.parent.parent / "wilor_pose_3d"

        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

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
            
            # 파일 이름을 기반으로 한 출력 경로 생성
            img_basename = os.path.basename(img_path)
            img_fn, _ = os.path.splitext(img_basename)
            file_basename = f"keypoint_{img_fn.split('_')[-1]}"

            # CSV 저장을 위한 경로 설정
            # 원본 이미지의 프레임 번호에 해당하는 폴더 구조를 유지 (예: processed_270_480)
            keypoint_save_dir = output_base_dir_root / "processed_270_480" / "keypoint"
            os.makedirs(keypoint_save_dir, exist_ok=True)
            csv_save_path = keypoint_save_dir / f"{file_basename}.csv"
            df = pd.DataFrame(all_keypoints, columns=['x', 'y', 'z'])
            df.to_csv(csv_save_path, index=False)

            # # 시각화 저장을 위한 경로 설정
            # visualize_save_dir = output_base_dir_root / "processed_270_480" / "visualize"
            # os.makedirs(visualize_save_dir, exist_ok=True)
            # png_save_path = visualize_save_dir / f"{file_basename}.png"
            
            # # 시각화 및 저장
            # fig = plt.figure(figsize=(10, 10))
            # ax = fig.add_subplot(111, projection='3d')
            
            # x, y, z = all_keypoints[:, 0], all_keypoints[:, 1], -all_keypoints[:, 2]
            # ax.scatter(x, y, z, c='r', s=40)
            
            # 오른손과 왼손 관절 그리기
            # for i, j in HAND_CONNECTIONS:
            #     ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='blue') # 오른손
            #     ax.plot([x[i+21], x[j+21]], [y[i+21], y[j+21]], [z[i+21], z[j+21]], color='green') # 왼손
                
            # ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            # ax.invert_yaxis(); ax.view_init(elev=10, azim=270)
            # plt.title(os.path.basename(png_save_path)); plt.tight_layout()
            # plt.savefig(png_save_path); plt.close(fig)

    print("\n모든 작업이 완료되었습니다.")


if __name__ == '__main__':
    process_and_visualize_all_folders()
