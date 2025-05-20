from imports import *

DATASET_NAME = "the-oxfordiiit-pet-dataset"

# 로깅 설정
def get_logger(log_level=logging.INFO):
    """로거 가져오기
    
    Args:
        log_level: 로깅 레벨 (기본값: logging.INFO)
        
    Returns:
        logging.Logger: 로거 객체
    """
    # 로거가 이미 초기화되었는지 확인
    logger = logging.getLogger()
    if logger.handlers:  # 이미 핸들러가 있으면 그대로 반환
        return logger
    else:  # 핸들러가 없는 경우 새로 설정
        return setup_logging(log_level)

def setup_logging(log_level=logging.INFO):
    """로깅 시스템 초기화 및 설정
    
    Args:
        log_level: 로깅 레벨 (기본값: logging.INFO)
    
    Returns:
        logging.Logger: 구성된 로거 객체
    """
    # 로그 디렉토리 경로 설정
    log_dir = os.path.join(BASE_DIR, 'logs')
    
    # 로그 디렉토리가 없으면 생성
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
            print(f"로그 디렉토리 생성 완료: {log_dir}")
        except Exception as e:
            print(f"로그 디렉토리 생성 실패: {e}. 현재 디렉토리에 로그 파일을 생성합니다.")
            log_dir = BASE_DIR  # 생성 실패시 현재 디렉토리 사용
    
    # 현재 시간을 이용해 로그 파일 이름 생성
    log_file = os.path.join(log_dir, f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 로그 형식 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 루트 로거 가져오기
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 이미 핸들러가 존재하는 경우 제거 (중복 방지)
    if logger.handlers:
        for handler in logger.handlers[:]:  # 목록의 복사본을 사용해 안전하게 제거
            logger.removeHandler(handler)
            handler.close()  # 핸들러 자원 해제 및 파일 닫기
    
    # 파일 핸들러 설정
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        print(f"로그 파일이 생성되었습니다: {log_file}")
    except Exception as e:
        print(f"로그 파일 핸들러 생성 오류: {e}")
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # 로그 파일 경로 출력
    logger.info(f"로그 파일 경로: {os.path.abspath(log_file)}")
    
    return logger

# 기본 로거 가져오기 (다른 파일에서 사용 가능)
def get_logger():
    """기본 로거 가져오기
    
    Returns:
        logging.Logger: 로거 객체
    """
    return logging.getLogger()

# 로깅 설정 초기화
logger = get_logger()

# 한글 폰트 설정 함수
def setup_korean_font():
    """matplotlib에서 한글 폰트를 사용할 수 있도록 설정"""
    if platform.system() == 'Darwin':  # macOS
        plt.rc('font', family='AppleGothic')
    elif platform.system() == 'Windows':  # Windows
        plt.rc('font', family='Malgun Gothic')
    else:  # Linux
        plt.rc('font', family='NanumGothic')
    
    # matplotlib의 기본 설정 변경
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    
    # 폰트 설정이 잘 되었는지 확인 및 디버깅 메시지 출력
    logger.info(f"현재 폰트 설정: {mpl.rcParams['font.family']}")
    
    # 사용 가능한 폰트 목록 중 일부만 출력
    fonts = [f.name for f in fm.fontManager.ttflist]
    logger.info(f"사용 가능한 폰트 중 일부: {fonts[:5]}...")

def setup_device():
    """
    CUDA(GPU) 또는 CPU 디바이스를 자동으로 선택

    Returns:
        torch.device: 사용할 계산 디바이스
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        cuda_device = 0  # 첫 번째 GPU 사용
        device = torch.device(f"cuda:{cuda_device}")
        logger.info(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}, 선택된 GPU: {cuda_device}")
        # GPU 정보 출력
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"GPU {i}: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA 사용 불가. CPU 사용")
    logger.info(f"선택된 장치: {device}")
    return device

# 현재 스크립트 위치 기준 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 프로젝트 루트 디렉토리

# 결과물 저장 경로 설정
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, 'visualizations')

# 데이터셋 경로 설정 (절대 경로 사용)
def __find_dataset_path__(dataset_name=None):
    """데이터셋 경로를 찾는 함수
    여러 가능한 위치를 검색하여 데이터셋 경로를 찾는다.
    
    Returns:
        str: 데이터셋의 절대 경로
    """
    possible_paths = [
        # 상대 경로
        os.path.join(BASE_DIR, f'../{dataset_name}'),
        os.path.join(PROJECT_ROOT, dataset_name),
    ]
    
    for path in possible_paths:
        # 경로 정규화 (상대 경로를 절대 경로로 변환)
        norm_path = os.path.normpath(os.path.abspath(path))
        
        # 해당 경로에 데이터셋이 있는지 확인 (trainval.txt 파일 확인)
        check_file = os.path.join(norm_path, 'annotations', 'annotations', 'trainval.txt')
        if os.path.exists(check_file):
            logger.info(f"데이터셋 경로 발견: {norm_path}")
            return norm_path
    
    # 경로를 찾지 못한 경우 기본값 반환 및 경고
    default_path = os.path.join(PROJECT_ROOT, dataset_name)
    logger.warning(f"경고: 데이터셋 경로를 찾을 수 없습니다. 기본값 사용: {default_path}")
    return default_path

# 데이터셋 경로 찾기
DATASET_PATH = __find_dataset_path__(dataset_name=DATASET_NAME)

# 파일 경로 설정
TRAINVAL_FILE_PATH = os.path.join(DATASET_PATH, "annotations", "annotations", "trainval.txt")
TEST_FILE_PATH = os.path.join(DATASET_PATH, "annotations", "annotations", "test.txt")
IMAGE_DIR = os.path.join(DATASET_PATH, "images", "images")
XML_DIR = os.path.join(DATASET_PATH, "annotations", "annotations", "xmls")

# 결과물 저장 디렉토리 생성 함수
def create_results_directories(custom_dir=None):
    """결과물 저장을 위한 디렉토리 구조 생성
    
    Args:
        custom_dir (str, optional): 사용자 지정 디렉토리. 설정 시 기본 경로 대신 사용
    
    Returns:
        str: 시각화 결과물이 저장될 디렉토리 경로
    """
    # 사용자 지정 디렉토리가 있는 경우 사용
    if custom_dir is not None:
        vis_dir = os.path.join(custom_dir, 'visualizations')
    else:
        # 기본 경로 사용
        os.makedirs(RESULTS_DIR, exist_ok=True)
        vis_dir = VISUALIZATION_DIR
    
    # 시각화 디렉토리 생성
    os.makedirs(vis_dir, exist_ok=True)
    
    logger.info(f"결과물 저장 디렉토리 준비 완료: {vis_dir}")
    return vis_dir

# XML 파일 이름 가져오기 (확장자 제거)
def __extract_xml_list__(xml_dir):
    xml_list = [os.path.splitext(file)[0] for file in os.listdir(xml_dir) if file.endswith(".xml")]
    return xml_list

# 이미지 데이터에 대해 XML 파일이 있는지 확인
def check_exist_format(image_list, xml_dir):
    xml_list = __extract_xml_list__(xml_dir)

    # 이미지에 대해 XML 파일이 없는 경우 확인
    missing_xml = [image for image in image_list if image not in xml_list]

    # 이미지에 대해 XML 파일이 있는 경우 확인
    image_list = [image for image in image_list if image in xml_list]

    # 결과 출력
    logger.warning(f"XML 파일이 없는 이미지 수: {len(missing_xml)}")
    logger.warning(missing_xml)
    
    return xml_list, image_list, missing_xml