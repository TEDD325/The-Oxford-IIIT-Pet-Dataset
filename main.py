# 메인 실행 파일

# 모든 의존성 라이브러리 및 모듈 가져오기
from imports import *
from config import *

from visualization import visualize_annotated_image, visualize_class_distribution, visualize_sample_images, visualize_predictions

# 데이터 및 모델 모듈 임포트
from data_utils import load_data, load_annotations, extract_xml_files, print_data_size_info, analyze_class_distribution, create_trainval_test_list, AnnotatedImageDataset, TestDataset, create_dataset
from dataset_utils import create_datasets, create_dataloaders
from model import initialize_model, train_model, test_model

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
    test_xml_list, _, missing_test_xml = check_exist_format(test_list, XML_DIR)
    ''' Test 셋에는 XML 파일이 없는 것이 당연 '''
    
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
    
    
    # 클래스 정보 정의
    classes = ["background", "dog", "cat"]

    # dataset_utils에서 추가한 함수를 활용하여 데이터셋 생성
    
    
    # 학습, 검증, 테스트 데이터셋 생성 
    train_dataset, valid_dataset, test_dataset = create_datasets(
        image_dir=IMAGE_DIR,
        trainval_list=trainval_list,
        test_list=test_list,
        xml_dir=XML_DIR,
        classes=classes,
        valid_size=0.3,
        random_state=42
    )

    # 데이터로더 생성
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        batch_size=8
    )
    
    # 메인 함수 인자 처리
    parser = argparse.ArgumentParser(description='학습 및 테스트 파라미터')
    parser.add_argument('--model_path', type=str, default=None, help='모델 파일 경로, 지정하지 않으면 처음부터 학습')
    parser.add_argument('--save_dir', type=str, default='models', help='모델 저장 디렉토리')
    parser.add_argument('--model_name', type=str, default='ssd_pet_detector', help='모델 파일 이름 접두사')
    parser.add_argument('--epochs', type=int, default=5, help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='학습률')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'train_test'], help='실행 모드')
    args = parser.parse_args()
    
    # 모델 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용하는 장치: {device}")
    
    # SSD 모델 초기화 (저장된 모델 불러오기 지원)
    model = initialize_model(classes=classes, device=device, model_path=args.model_path)
    
    # 실행 모드 검사
    if args.mode in ['train', 'train_test']:
        # 모델 학습
        logger.info(f"모델 학습 시작: {args.epochs} 에폭, 학습률 {args.learning_rate}, 배치 크기 {args.batch_size}")
        training_results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            classes=classes,
            device=device,
            num_epochs=args.epochs,  
            lr=args.learning_rate,
            save_dir=args.save_dir,
            model_name=args.model_name
        )
        
        logger.info(f"학습 결과: 최고 mAP = {training_results['best_map']:.4f}")
        
        # 학습 결과 저장 (JSON 파일)
        
        results_path = os.path.join(args.save_dir, f'{args.model_name}_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_map': float(training_results['best_map']),
                'best_epoch': training_results['best_epoch'],
                'best_model_path': training_results['best_model_path'],
                'last_model_path': training_results['last_model_path']
            }, f, ensure_ascii=False, indent=4)
        logger.info(f"학습 결과 저장 완료: {results_path}")
    
    if args.mode in ['test', 'train_test']:
        # 테스트 실행
        logger.info("테스트 시작...")
        test_results = test_model(
            model=model,
            test_loader=test_loader,
            classes=classes,
            device=device
        )
        
        logger.info(f"테스트 완료: {len(test_results['predictions'])}개 이미지 추론")
        
        # 예측 결과 시각화 및 저장
        logger.info("예측 결과 시각화 시작...")
        
        # 테스트 이미지 중 일부만 시각화 (기본 5개)
        prediction_files = visualize_predictions(
            images=test_results['images'],
            predictions=test_results['predictions'],
            classes=classes,
            save_dir=os.path.join('results', 'predictions'),
            score_threshold=0.5,  # 객체 검출 임계값
            max_images=5  # 시각화할 최대 이미지 수
        )
        
        if prediction_files:
            logger.info(f"예측 결과 이미지 {len(prediction_files)}개 저장 완료")
        else:
            logger.warning("예측 결과 이미지 저장 실패")


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
