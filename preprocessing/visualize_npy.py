import numpy as np
import matplotlib.pyplot as plt
import os

# --- ⚙️ 시각화할 파일 설정 ⚙️ ---
# 여기에 방금 생성된 .npy 파일의 전체 경로를 입력하세요.
npy_file_path = "/media/kcy/3A72CA7272CA3285/data_MHAV/assemble/hg/assemble_hexNut-metalBlock3-spacer-hexBolt_hand_hg/hg_assemble_hexNut-metalBlock3-spacer-hexBolt_hand_hg_wrist_turn.npy"
# --------------------------------

try:
    # 1. .npy 파일 불러오기
    rotation_data = np.load(npy_file_path)

    # 2. 그래프 그리기
    plt.figure(figsize=(12, 6))  # 그래프 크기 설정
    plt.plot(rotation_data, label='Cumulative Rotation')

    # 3. 그래프 꾸미기
    plt.title(f"wrist Feature Visualization\n({os.path.basename(npy_file_path)})")
    plt.xlabel("Frame Index")
    plt.ylabel("Cumulative Rotation Angle (degrees)")
    plt.grid(True)  # 그리드 표시
    plt.legend()
    plt.tight_layout() # 레이아웃 최적화

    # 4. 그래프 창 보여주기
    plt.show()

except FileNotFoundError:
    print(f"Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {npy_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")