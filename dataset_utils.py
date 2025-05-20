from imports import *
from config import *
from data_utils import create_dataset, AnnotatedImageDataset, TestDataset
from sklearn.model_selection import train_test_split

# 로거 가져오기
logger = get_logger()


def create_datasets(image_dir, trainval_list, test_list, xml_dir, classes, valid_size=0.3, random_state=42):
    """
    학습, 검증, 테스트 데이터셋을 한 번에 생성합니다.
    클래스 불균형을 고려한 층화 샘플링(stratified sampling)을 수행합니다.
    
    Args:
        image_dir: 이미지 파일이 저장된 디렉토리 경로
        trainval_list: 학습 및 검증용 이미지 리스트
        test_list: 테스트용 이미지 리스트
        xml_dir: XML 어노테이션 디렉토리 경로
        classes: 클래스 이름 리스트
        valid_size: 검증 데이터셋으로 나눌 비율 (0~1 사이)
        random_state: 난수 시드 값
    
    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset) 형태의 데이터셋 튜플
    """
    # trainval_list가 단순 리스트인지 데이터프레임인지 확인
    import pandas as pd
    
    if isinstance(trainval_list, pd.DataFrame):
        # 이미 데이터프레임인 경우, Species 열을 stratify로 사용
        stratify_values = trainval_list['Species'] if 'Species' in trainval_list.columns else None
        train_set, valid_set = train_test_split(
            trainval_list, 
            test_size=valid_size, 
            random_state=random_state,
            stratify=stratify_values
        )
    else:
        # 단순 리스트인 경우 stratify 없이 분할
        logger.warning("trainval_list가 데이터프레임이 아니어서 stratify 없이 train_test_split 수행")
        train_set, valid_set = train_test_split(
            trainval_list, 
            test_size=valid_size, 
            random_state=random_state
        )
    
    # 학습 데이터셋 생성
    train_dataset = create_dataset(
        image_dir=image_dir,
        image_list=train_set,
        dataset_type="train",
        annotation_dir=xml_dir,
        classes=classes
    )
    
    # 검증 데이터셋 생성
    valid_dataset = create_dataset(
        image_dir=image_dir,
        image_list=valid_set,
        dataset_type="val",
        annotation_dir=xml_dir,
        classes=classes
    )
    
    # 테스트 데이터셋 생성
    test_dataset = create_dataset(
        image_dir=image_dir,
        image_list=test_list,
        dataset_type="test"
    )
    
    # 데이터셋 크기 로깅
    logger.info(f"데이터셋 생성 완료")
    
    return train_dataset, valid_dataset, test_dataset


def create_dataloaders(train_dataset, valid_dataset, test_dataset=None, batch_size=8):
    """
    학습, 검증, 테스트 데이터로더를 한 번에 생성합니다.
    
    Args:
        train_dataset: 학습 데이터셋
        valid_dataset: 검증 데이터셋
        test_dataset: 테스트 데이터셋 (선택사항)
        batch_size: 배치 크기
    
    Returns:
        tuple: 데이터로더 튜플 (train_loader, valid_loader) 또는 
               (train_loader, valid_loader, test_loader)
    """
    # 콜레이트 함수 정의 (배치 처리를 위한 데이터 포맷 변환)
    collate_fn = lambda x: tuple(zip(*x))
    
    # 학습 및 검증 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    logger.info(f"Train 데이터셋 크기: {len(train_dataset)}")
    logger.info(f"Validation 데이터셋 크기: {len(valid_dataset)}")
    
    # 테스트 데이터 로더 생성 (있는 경우에만)
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        logger.info(f"Test 데이터셋 크기: {len(test_dataset)}")

        return train_loader, valid_loader, test_loader
    
    return train_loader, valid_loader
