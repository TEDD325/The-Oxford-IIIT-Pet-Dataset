# 데이터 로딩 및 전처리

from imports import *
from config import *
import json
import hashlib

# BASE_DIR 가져오기 (캐시 디렉토리 생성에 사용)
from config import BASE_DIR

# 로거 가져오기
logger = get_logger()

def load_data(sep="\s+", columns=None):
    """파일에서 훈련/검증 및 테스트 데이터를 로드"""
    # 훈련/검증 파일 읽기
    df_trainval = pd.read_csv(TRAINVAL_FILE_PATH, sep=sep, header=None)
    df_trainval.columns = columns

    # 테스트 파일 읽기
    df_test = pd.read_csv(TEST_FILE_PATH, sep=sep, header=None)
    df_test.columns = columns
    
    return df_trainval, df_test

def extract_xml_files(annotation_file_format):
    xml_files = [file for file in os.listdir(XML_DIR) if file.endswith(annotation_file_format)]
    logger.info(f"XML 파일 개수: {len(xml_files)}")
    return xml_files

def print_data_size_info(df_trainval, df_test, annotation_file_format):

    logger.info(f"훈련/검증 데이터 수: {len(df_trainval)}")
    logger.info(f"테스트 데이터 수: {len(df_test)}")

def analyze_class_distribution(df_trainval, df_test, column):
    """클래스 불균형 분석"""
    logger.info("클래스 불균형 확인:")
    trainval_class_counts = df_trainval[column].value_counts()
    test_class_counts = df_test[column].value_counts()

    logger.info("훈련/검증 데이터셋:")
    for species, count in trainval_class_counts.items():
        ratio = (count / len(df_trainval)) * 100
        logger.info(f"- {column}: {species}: {count}개 ({ratio:.2f}%)")

    logger.info("테스트 데이터셋:")
    for species, count in test_class_counts.items():
        ratio = (count / len(df_test)) * 100
        logger.info(f"- {column}: {species}: {count}개 ({ratio:.2f}%)")
        
    return trainval_class_counts, test_class_counts

# TODO: 쓰이지 않는다면 제거를 고려
def create_trainval_test_list(df_trainval, df_test, column):
    """Train과 Validation, Test에 사용될 이미지 파일 이름 리스트 생성"""
    trainval_list = df_trainval[column].tolist()
    test_list = df_test[column].tolist()
    return trainval_list, test_list

# """
# # XML 파일 이름 가져오기 (확장자 제거)
# xml_list = [os.path.splitext(file)[0] for file in os.listdir(xml_dir) if file.endswith(".xml")]

# # Train 이미지에 대해 XML 파일이 없는 경우 확인
# missing_xml = [image for image in trainval_list if image not in xml_list]

# # Train 이미지에 대해 XML 파일이 있는 경우 확인
# trainval_list = [image for image in trainval_list if image in xml_list]

# # 결과 출력
# print(f"XML 파일이 없는 Train 이미지 수: {len(missing_xml)}")
# print(missing_xml)
# """
# def __extract_xml_list__(xml_dir):
#     xml_list = [os.path.splitext(file)[0] for file in os.listdir(xml_dir) if file.endswith(".xml")]
#     return xml_list

# def check_exist_xml(dataset_list, xml_dir):
#     xml_list = __extract_xml_list__(xml_dir)
#     missing_xml = [image for image in dataset_list if image not in xml_list]
#     dataset_list = [image for image in dataset_list if image in xml_list]
#     return dataset_list, missing_xml


# XML 어노테이션 로드 및 파싱
def load_annotations(xml_dir, xml_files, cache_dir=None):
    """파일에서 애노테이션 로드 또는 XML 파싱 수행
    
    저장된 애노테이션 파일이 있으면 로드하고, 없으면 XML 파싱 수행
    
    Args:
        xml_dir (str): XML 파일이 있는 디렉토리 경로
        xml_files (list): 처리할 XML 파일 리스트
        cache_dir (str, optional): 캐시 파일 저장 디렉토리. 기본값은 None이며, 이 경우 BASE_DIR/cache 디렉토리를 사용
    
    Returns:
        list: 애노테이션 리스트
    """
    # 캐시 디렉토리 설정
    if cache_dir is None:
        cache_dir = os.path.join(BASE_DIR, 'cache')
    
    # 캐시 디렉토리가 없으면 생성
    os.makedirs(cache_dir, exist_ok=True)
    
    # 캐시 파일 경로 생성
    # 해시값을 활용하여 파일 목록으로부터 유니크한 식별자 생성
    files_hash = hashlib.md5(''.join(sorted(xml_files)).encode()).hexdigest()[:10]
    cache_filename = f"annotations_cache_{len(xml_files)}_{files_hash}.json"
    cache_file = os.path.join(cache_dir, cache_filename)
    
    # 캐시 파일이 존재하면 로드
    if os.path.exists(cache_file):
        logger.info(f"저장된 애노테이션 파일 발견: {cache_file} - 로드를 시도합니다.")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            logger.info(f"애노테이션 로드 성공! {len(annotations)}개의 애노테이션을 불러왔습니다.")
            return annotations
        except Exception as e:
            logger.warning(f"애노테이션 캐시 파일 로드 실패: {e} - XML 파싱으로 전환합니다.")
    
    # 캐시 파일이 없거나 로드 실패시 XML 파싱 수행
    logger.info(f"XML 파일 파싱을 수행합니다...")
    annotations = __parse_xml__(xml_dir, xml_files)
    
    # 파싱 결과를 캐시 파일로 저장
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        logger.info(f"애노테이션을 캐시파일로 저장했습니다: {cache_file}")
    except Exception as e:
        logger.warning(f"애노테이션 캐시 파일 저장 실패: {e}")
    
    return annotations

# XML에서 객체 정보 추출 (내부용 유틸리티 함수)
def __extract_object_from_xml__(obj, width=0, height=0):
    """XML의 object 요소에서 객체 정보(이름, 바운딩 박스)를 추출
    
    Args:
        obj: XML의 object 요소
        width: 이미지 너비 (0이면 범위 검사 안함)
        height: 이미지 높이 (0이면 범위 검사 안함)
    
    Returns:
        dict: 객체 이름과 바운딩 박스 정보를 포함하는 사전, 실패 시 None
    """
    try:
        obj_name = obj.find('name').text
        bbox = obj.find('bndbox')
        
        x_min = int(float(bbox.find('xmin').text))
        y_min = int(float(bbox.find('ymin').text))
        x_max = int(float(bbox.find('xmax').text))
        y_max = int(float(bbox.find('ymax').text))
        
        # 경계 상자가 이미지 크기를 벗어나지 않도록 조정 (0 이상, 이미지 크기 이하)
        if width > 0 and height > 0:
            x_min = max(0, min(x_min, width))
            y_min = max(0, min(y_min, height))
            x_max = max(0, min(x_max, width))
            y_max = max(0, min(y_max, height))
        
        # 유효한 영역인지 확인 (최소한 1픽셀 이상의 너비와 높이)
        if x_max > x_min and y_max > y_min:
            return {
                'name': obj_name,
                'bbox': [x_min, y_min, x_max, y_max]
            }
        else:
            return None
    except (AttributeError, ValueError) as e:
        return None

# XML 파싱 (내부용 함수)
def __parse_xml__(xml_dir, xml_files):
    """ XML 구조
    annotation: None
        folder: OXIIIT
        filename: Abyssinian_1.jpg
        source: None
            database: OXFORD-IIIT Pet Dataset
            annotation: OXIIIT
            image: flickr
        size: None
            width: 800
            height: 419
            depth: 3
        segmented: 0
        object: None
            name: Abyssinian
            pose: Frontal
            truncated: 0
            difficult: 0
            bndbox: None
                xmin: 106
                ymin: 27
                xmax: 688
                ymax: 348
    """
    annotations = []
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 파일명 추출
            try:
                filename = root.find('filename').text
            except (AttributeError, ValueError) as e:
                logger.warning(f"파일명 요소 누락 또는 오류 (파일: {xml_file}): {e}")
                filename = xml_file.replace('.xml', '.jpg')  # 기본값으로 XML 파일명 사용
            
            # 크기 정보 추출
            try:
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            except (AttributeError, ValueError) as e:
                logger.warning(f"크기 정보 요소 누락 또는 오류 (파일: {xml_file}): {e}")
                width, height = 0, 0  # 기본값
            
            # 객체 정보 추출
            objects = []
            for obj in root.findall('object'):
                object_info = __extract_object_from_xml__(obj, width, height)
                if object_info:
                    objects.append(object_info)
                else:
                    logger.warning(f"객체 정보 추출 오류 (파일: {xml_file})")
            
            # 어노테이션 정보 저장
            annotation = {
                'filename': filename,
                'width': width,
                'height': height,
                'objects': objects
            }
            
            annotations.append(annotation)

        except Exception as e:
            logger.warning(f"XML 파일 파싱 오류 (파일: {xml_file}): {e}")
            continue
    
    logger.info(f"XML 파일 파싱 완료: 총 {len(xml_files)}개 중 {len(annotations)}개 처리")
    
    return annotations



def __transform_image__(dtype=torch.float32, scale=True):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=dtype, scale=scale),
        ]
    )

    return transform



class TestDataset(Dataset):
    def __init__(self, image_dir, image_list, transforms=None, image_format=".jpg", color_mode="RGB"):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_files = image_list  # 테스트 이미지 리스트 (확장자 없음)
        self.image_format = image_format
        self.color_mode = color_mode

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 파일 경로
        image_file = self.image_files[idx] + self.image_format
        image_path = os.path.join(self.image_dir, image_file)

        # 이미지 로드
        image = Image.open(image_path).convert(self.color_mode)

        # Transform 적용
        if self.transforms:
            image = self.transforms(image)

        return image, self.image_files[idx]  # 이미지와 파일 이름 반환

    """
        Q. 테스트 데이터셋을 별도로 생성하는 이유?
        A. 테스트 데이터셋은 추론 시에 사용되므로 레이블 정보가 없을 수 있다.
    """


class AnnotatedImageDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, classes, image_list, transforms=None, image_format=".jpg", annotation_format=".xml", color_mode="RGB"):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.classes = classes
        self.transforms = transforms
        self.image_files = image_list # 미리 필터링된 유효한 이미지 파일 리스트
        self.image_format = image_format
        self.annotation_format = annotation_format
        self.color_mode = color_mode

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 및 XML 파일 경로 설정
        image_file = self.image_files[idx] + self.image_format
        annotation_file = self.image_files[idx] + self.annotation_format
        image_path = os.path.join(self.image_dir, image_file)
        annotation_path = os.path.join(self.annotation_dir, annotation_file)

        # 이미지 로드
        image = Image.open(image_path).convert(self.color_mode)

        # 어노테이션 로드
        boxes = []
        labels = []
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # 이미지 크기 추출 (바운딩 박스 검증에 사용)
        try:
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        except (AttributeError, ValueError):
            # 이미지에서 크기 추출
            width, height = image.size

        # 공통 추출 함수를 사용하여 객체 정보 추출
        for obj in root.findall("object"):
            object_info = __extract_object_from_xml__(obj, width, height)
            if object_info and object_info['name'] in self.classes:
                class_name = object_info['name']
                labels.append(self.classes.index(class_name))
                boxes.append(object_info['bbox'])

        # Tensor로 변환
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) # TODO: int32 타입으로 한 번 시도해보고 런타임 오류가 뜨는지 직접 확인해 볼 필요가 있다.
        """
            Q. labels = torch.tensor(labels, dtype=torch.int64)에서 dtype=torch.int64이어야 하는 이유는?
            A.
                1. int64 타입만 써야 하는 것은 아니지만, PyTorch 생태계 내에서 호환성을 위해 일반적으로 이 타입을 사용한다. 
                2. nn.CrossEntropyLoss와 같은 분류 손실 함수는 기본적으로 torch.int64 타입의 라벨을 기대한다. 다른 타입으로 지정하면 런타임 오류가 발생할 수 있다.
                3. 메모리 사용량이 매우 중요한 대규모 데이터셋의 경우 때로는 torch.int32와 같은 작은 타입을 사용할 수도 있지만, 그럴 경우 다른 함수들과 호환되도록 명시적인 타입 변환이 필요할 수 있다.
        """

        # Transform 적용
        if self.transforms:
            image, boxes, labels = self.transforms(image, boxes, labels)

        target = {"boxes": boxes, "labels": labels}
        return image, target


def create_dataset(
    image_dir,
    image_list,
    dataset_type="train",
    annotation_dir=None,
    classes=None,
    apply_transforms=True,
    image_format=".jpg",
    annotation_format=".xml",
    color_mode="RGB"
):
    """
    정해진 파라미터를 기반으로 트레이닝/검증 또는 테스트 데이터셋을 생성합니다.
    
    Args:
        image_dir: 이미지 파일이 저장된 디렉토리
        image_list: 이미지 파일 리스트 (확장자 없음)
        dataset_type: 데이터셋 유형, 'train' 또는 'test'
        annotation_dir: XML 파일이 저장된 디렉토리 (dataset_type='train'인 경우만 필요)
        classes: 클래스 이름 리스트 (dataset_type='train'인 경우만 필요)
        apply_transforms: 이미지 변환 함수 적용 여부
        image_format: 이미지 파일 확장자
        annotation_format: XML 파일 확장자
        color_mode: 이미지 색상 모드
    
    Returns:
        Dataset: VOCDataset 또는 TestDataset 인스턴스
    """

    if apply_transforms:
        transforms = __transform_image__()
    else:
        transforms = None
    
    # 데이터셋 유형에 따라 다른 클래스 사용
    if dataset_type.lower() == "train" or dataset_type.lower() == "val" or dataset_type.lower() == "valid":
        # 어노테이션과 클래스 정보가 있는지 확인
        if annotation_dir is None or classes is None:
            raise ValueError("Train/Validation 데이터셋은 annotation_dir와 classes가 필요합니다.")
        
        # AnnotatedImageDataset 생성
        dataset = AnnotatedImageDataset(
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            classes=classes,
            image_list=image_list,
            transforms=transforms,
            image_format=image_format,
            annotation_format=annotation_format,
            color_mode=color_mode
        )
    elif dataset_type.lower() == "test":
        # TestDataset 생성
        dataset = TestDataset(
            image_dir=image_dir,
            image_list=image_list,
            transforms=transforms,
            image_format=image_format,
            color_mode=color_mode
        )
    else:
        raise ValueError(f"지원되지 않는 dataset_type: {dataset_type}. 'train', 'val' 또는 'test'를 사용하세요.")
    
    return dataset
