import os
import shutil
import random
from pathlib import Path


def split_dataset(
    archive_dir="archive",
    data_dir="data",
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    seed=42
):
    """
    archive 폴더의 파일들을 train/val/test로 나누어 data 폴더로 이동
    
    Args:
        archive_dir: 원본 파일들이 있는 디렉토리
        data_dir: 목적지 디렉토리
        train_ratio: 훈련 데이터 비율 
        val_ratio: 검증 데이터 비율 
        test_ratio: 테스트 데이터 비율 
        seed: 랜덤 시드
    """

    random.seed(seed)

    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratio must sum to 1.0"
    
    archive_path = Path(archive_dir)
    data_path = Path(data_dir)
    
    if not archive_path.exists():
        raise FileNotFoundError(f"{archive_dir} 폴더를 찾을 수 없습니다.")
    
    classes = ["FAKE", "REAL"]
    splits = ["train", "val", "test"]
    
    for split in splits:
        for cls in classes:
            target_dir = data_path / split / cls
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"디렉토리 생성: {target_dir}")
    
    for cls in classes:
        source_dir = archive_path / cls
        
        all_files = [f for f in source_dir.iterdir() if f.is_file()]
        
        total_files = len(all_files)
        print(f"\n{cls} 클래스: 총 {total_files}개 파일 발견")
        
        random.shuffle(all_files)
        
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        
        train_files = all_files[:train_end]
        val_files = all_files[train_end:val_end]
        test_files = all_files[val_end:]    
        
        print(f"  - Train: {len(train_files)}개")
        print(f"  - Val: {len(val_files)}개")
        print(f"  - Test: {len(test_files)}개")
        
        file_splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        
        for split_name, files in file_splits.items():
            target_dir = data_path / split_name / cls
            
            for i, file_path in enumerate(files, 1):
                target_path = target_dir / file_path.name
                shutil.copy2(file_path, target_path)
                
                if i % 100 == 0:
                    print(f"  {split_name}/{cls}: {i}/{len(files)} 복사 완료")
            
            print(f"  {split_name}/{cls}: 모든 파일 복사 완료 ({len(files)}개)")
    
    print("\n데이터셋 분할 완료")
    print(f"결과 디렉토리: {data_path.absolute()}")


if __name__ == "__main__":
    split_dataset(
        archive_dir="archive",
        data_dir="data",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
    )
