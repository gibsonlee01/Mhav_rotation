from pathlib import Path
import shutil

def delete_target_folders(start_path_str: str, target_name: str):
    """
    지정된 경로(start_path_str)와 그 모든 하위 경로를 검색하여,
    이름이 target_name과 일치하는 모든 폴더를 찾아 삭제합니다.
    """
    try:
        start_path = Path(start_path_str).resolve() # 경로를 절대 경로로 변환
    except FileNotFoundError:
        print(f"오류: 시작 경로 '{start_path_str}'를 찾을 수 없습니다.")
        return

    if not start_path.is_dir():
        print(f"오류: 시작 경로 '{start_path}'가 폴더가 아닙니다.")
        return

    print(f"'{start_path}'에서 '{target_name}' 폴더를 검색하여 삭제를 시작합니다...")

    # .rglob()을 사용해 모든 하위 폴더와 파일을 재귀적으로 검색
    # 이름이 target_name과 일치하는 모든 항목을 찾습니다.
    for path_obj in start_path.rglob(target_name):
        # 이름이 정확히 일치하고, '폴더'인 경우에만 삭제 진행
        if path_obj.name == target_name and path_obj.is_dir():
            try:
                print(f"삭제 중: {path_obj}")
                # shutil.rmtree()는 폴더와 그 안의 모든 내용(하위 폴더, 파일)을 삭제합니다.
                shutil.rmtree(path_obj)
                print(f"✅ 삭제 완료: {path_obj}")
            except OSError as e:
                print(f"❌ 삭제 실패: {path_obj} ({e})")

    print("\n모든 작업이 완료되었습니다.")


if __name__ == '__main__':
    # 🚨 중요: 여기에 검색을 시작할 최상위 폴더 경로를 정확히 입력하세요.
    main_directory = '/media/kcy/3A72CA7272CA3285/data_MHAV' # 예시 경로입니다. 실제 경로로 변경하세요.

    # 🚨 중요: 삭제할 폴더의 정확한 이름을 입력하세요.
    folder_to_delete = 'Thermal_threashold'

    # 사용자에게 최종 확인을 받아 실수를 방지합니다.
    user_input = input(
        f"\n경고: '{main_directory}'와 그 하위 모든 폴더에서\n"
        f"'{folder_to_delete}'라는 이름의 폴더를 영구적으로 삭제합니다.\n"
        f"이 작업은 되돌릴 수 없습니다. 정말로 계속하시겠습니까? (yes/no): "
    )

    if user_input.lower() == 'yes':
        delete_target_folders(main_directory, folder_to_delete)
    else:
        print("작업이 취소되었습니다.")