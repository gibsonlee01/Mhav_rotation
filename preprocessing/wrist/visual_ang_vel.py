import numpy as np
import matplotlib.pyplot as plt
import os

def extract_rotational_features(trajectory: np.ndarray, threshold_ratio=0.2):
    """
    접근/후퇴 구간을 고려하여 더 정확한 회전 중심을 찾아 특징을 계산합니다.
    
    Args:
        trajectory (np.ndarray): (N, 2) 형태의 (x, y) 좌표 시퀀스.
        threshold_ratio (float): 각속도 상위 %를 핵심 동작으로 간주할지 비율.

    Returns:
        tuple: (각속도, 누적 각도, 계산된 중심점) 튜플.
    """
    # 1. 먼저 전체 데이터를 기반으로 '대략적인' 각속도를 계산
    temp_center = np.mean(trajectory, axis=0)
    relative_coords_temp = trajectory - temp_center
    angles_temp = np.arctan2(relative_coords_temp[:, 1], relative_coords_temp[:, 0])
    angular_velocity_temp = np.diff(angles_temp)
    angular_velocity_temp = (angular_velocity_temp + np.pi) % (2 * np.pi) - np.pi
    
    # 2. '핵심 동작 구간' 식별
    abs_ang_vel = np.abs(angular_velocity_temp)
    threshold = np.quantile(abs_ang_vel, 1.0 - threshold_ratio)
    core_motion_indices = np.where(abs_ang_vel > threshold)[0]
    
    # 3. '핵심 동작 구간'의 좌표만으로 '진짜' 중심점 계산
    if len(core_motion_indices) > 0:
        core_trajectory = trajectory[core_motion_indices]
        final_center = np.mean(core_trajectory, axis=0)
    else:
        final_center = temp_center

    # 4. '진짜' 중심점으로 최종 특징들을 다시 계산
    relative_coords_final = trajectory - final_center
    angles_final = np.arctan2(relative_coords_final[:, 1], relative_coords_final[:, 0])
    angular_velocity_raw = np.diff(angles_final)
    angular_velocity = (angular_velocity_raw + np.pi) % (2 * np.pi) - np.pi
    angular_velocity = np.insert(angular_velocity, 0, 0)
    cumulative_angle = np.cumsum(angular_velocity)
    
    return angular_velocity.reshape(-1, 1), cumulative_angle.reshape(-1, 1), final_center

# --- ⚙️ 메인 코드 ---

# 1. 여기에 분석할 NPY 파일 경로를 입력하세요.
npy_file_path = '/media/kcy/3A72CA7272CA3285/data_MHAV/screw/sp/screw_woodChunk-socketCapScrew_allenKey_sp/sp_screw_woodChunk-socketCapScrew_allenKey_sp_wrist_trajectory.npy' 
npy_file_path = '/media/kcy/3A72CA7272CA3285/data_MHAV/unscrew/hg/unscrew_woodChunk-socketCapScrew_allenKey_hg/hg_unscrew_woodChunk-socketCapScrew_allenKey_hg_wrist_trajectory.npy' 

# 2. NPY 파일 로드 및 특징 추출
try:
    wrist_trajectory = np.load(npy_file_path)
    if wrist_trajectory.ndim != 2 or wrist_trajectory.shape[1] != 2:
        raise ValueError("NPY 파일은 (N, 2) 형태의 배열이어야 합니다.")
    
    # --- ⭐️ 수정된 부분 1: 3개의 변수로 반환값을 받습니다 ---
    angular_vel, cumulative_ang, final_center = extract_rotational_features(wrist_trajectory, 0.4)

    print(f"'{os.path.basename(npy_file_path)}' 파일 로드 성공!")
    print(f"데이터 형태: {wrist_trajectory.shape}")
    print(f"계산된 최종 중심점: {final_center}")

    # 3. 결과 시각화
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f'Rotational Feature Analysis\n(File: {os.path.basename(npy_file_path)})', fontsize=16)

    # Plot 1: 원본 궤적
    axes[0].plot(wrist_trajectory[:, 0], wrist_trajectory[:, 1], '.-', color='royalblue', label='Trajectory')
    # --- ⭐️ 수정된 부분 2: 계산된 최종 중심점을 'X'로 표시 ---
    axes[0].scatter(final_center[0], final_center[1], c='red', marker='x', s=100, label='Calculated Center', zorder=5)
    axes[0].set_title('Original Wrist Trajectory', fontsize=12)
    axes[0].set_xlabel('X-coordinate')
    axes[0].set_ylabel('Y-coordinate')
    axes[0].grid(True)
    axes[0].axis('equal')
    axes[0].legend()

    # Plot 2: 각속도 (회전 방향)
    axes[1].plot(angular_vel, color='forestgreen')
    axes[1].axhline(0, color='r', linestyle='--', linewidth=1)
    axes[1].set_title('Angular Velocity (Direction)', fontsize=12)
    axes[1].set_ylabel('Radians / frame')
    axes[1].grid(True)

    # Plot 3: 누적 각도 (총 회전량)
    axes[2].plot(cumulative_ang, color='darkorange')
    axes[2].set_title('Cumulative Angle (Total Rotation)', fontsize=12)
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Total Radians')
    axes[2].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

except FileNotFoundError:
    print(f"오류: '{npy_file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"오류 발생: {e}")