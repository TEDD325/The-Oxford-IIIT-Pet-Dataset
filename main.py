# 메인 실행 파일

# 분리된 모듈 가져오기
from imports import *
from config import *
from data_utils import *
from visualization import *

# 로깅 초기화 및 로거 가져오기
# 모듈 로드 시 실행하지 않고, 프로그램 실행 시작점에 로깅을 초기화하도록 변경


def main():
    """메인 함수: 프로젝트의 전체 실행 흐름을 제어합니다."""
    # 로깅 초기화
    global logger
    logger = get_logger()
    
    # 한글 폰트 설정
    setup_korean_font()
    
    # 장치 설정
    device = setup_device()
    
    # 데이터 로딩
    df_trainval, df_test = load_data(sep='\s+', columns=["Image", "ClassID", "Species", "BreedID"])
    
    # 데이터 정보 출력
    print_data_size_info(df_trainval, df_test, annotation_file_format=".xml")
    
    # 클래스 불균형 분석
    trainval_class_counts, test_class_counts = analyze_class_distribution(df_trainval, df_test, column="Species")
    
    # 클래스 분포 시각화
    visualize_class_distribution(df_trainval, df_test)
    
    # 샘플 이미지 시각화
    visualize_sample_images(df_trainval, IMAGE_DIR)

    # Train/Validation, Test 이미지 파일 이름 리스트 생성
    trainval_list, test_list = create_trainval_test_list(df_trainval, df_test, column="Image")
    
    # Train/Validation 이미지에 대해 XML 파일이 있는지 확인
    trainval_xml_list, trainval_list, missing_trainval_xml = check_exist_format(trainval_list, XML_DIR)
    
    # Test 이미지에 대해 XML 파일이 있는지 확인
    test_xml_list, test_list, missing_test_xml = check_exist_format(test_list, XML_DIR)
    
    # XML 파일 이름 리스트 추출
    extracted_xml_files_list = extract_xml_files(annotation_file_format=".xml")

    # 애노테이션 로드 또는 파싱
    annotations = load_annotations(XML_DIR, extracted_xml_files_list)

    # 애노테이션 이미지 시각화 - 랜덤 이미지 선택
    visualize_annotated_image(df_trainval, annotations, IMAGE_DIR)
    
    # 고양이 이미지 시각화 (Species=1)
    visualize_annotated_image(df_trainval, annotations, IMAGE_DIR, species=1)
    
    # 개 이미지 시각화 (Species=2)
    visualize_annotated_image(df_trainval, annotations, IMAGE_DIR, species=2)
    
    
    # Train/Validation 분리 (trainval_list에서 80% Train, 20% Validation으로 나눔)
    train_set, valid_set = train_test_split(trainval_list, test_size=0.3, random_state=42)

    # Train Dataset
    train_dataset = create_dataset(
        image_dir=IMAGE_DIR, 
        annotation_dir=XML_DIR, 
        classes=["background", "dog", "cat"], 
        image_list=train_set)

    # Validation Dataset
    valid_dataset = create_dataset(
        image_dir=IMAGE_DIR, 
        annotation_dir=XML_DIR, 
        classes=["background", "dog", "cat"], 
        image_list=valid_set)

    # # Test Dataset
    # test_dataset = create_dataset(
    #     image_dir=IMAGE_DIR, 
    #     annotation_dir=XML_DIR, 
    #     classes=["background", "dog", "cat"], 
    #     image_list=test_set)


    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 데이터 크기 출력
    print(f"Train 데이터셋 크기: {len(train_dataset)}")
    print(f"Validation 데이터셋 크기: {len(valid_dataset)}")
    # print(f"Test 데이터셋 크기: {len(test_dataset)}")


if __name__ == "__main__":
    # 로깅 설정 초기화
    setup_logging()
    logger.info("프로그램 실행 시작")
    
    try:
        main()
        logger.info("프로그램 정상 종료")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        raise
