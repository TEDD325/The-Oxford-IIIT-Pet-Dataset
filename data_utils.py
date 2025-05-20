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

    # Annotation 개수 확인
    # xml_files = [file for file in os.listdir(XML_DIR) if file.endswith(annotation_file_format)]
    # logger.info(f"XML 파일 개수: {len(xml_files)}")

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
            width: 600
            height: 400
            depth: 3
        segmented: 0
        object: None
            name: cat
            pose: Frontal
            truncated: 0
            occluded: 0
            bndbox: None
                xmin: 333
                ymin: 72
                xmax: 425
                ymax: 158
            difficult: 0
    
    Args:
        xml_dir: XML 파일이 있는 디렉토리 경로
        xml_files: 처리할 XML 파일 리스트
    
    Returns:
        list: 애노테이션 리스트 (이미지 이름, 클래스, 바운딩 박스 정보)
    """
    annotations = []
    invalid_files = []
    no_objects_files = []
    invalid_objects = []
    
    total_files = len(xml_files)
    processed_files = 0
    
    logger.info(f"XML 파일 {total_files}개 파싱 시작")
    
    # 각 XML 파일에 대해 파싱 시도
    for xml_file in xml_files:
        try:
            xml_path = os.path.join(xml_dir, xml_file)
            
            # XML 파일 파싱 시도
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                logger.error(f"XML 파일 파싱 오류 ({xml_file}): {e}")
                invalid_files.append(xml_file)
                continue
            
            # 파일이름 필드 확인
            filename_elem = root.find("filename")
            if filename_elem is None or not filename_elem.text:
                logger.warning(f"XML 파일에 filename 요소가 없음: {xml_file}")
                image_name = os.path.splitext(xml_file)[0] + ".jpg"  # 기본값 사용
            else:
                image_name = filename_elem.text
            
            # object 요소들 확인
            objects = root.findall("object")
            if not objects:
                logger.warning(f"XML 파일에 object 요소가 없음: {xml_file}")
                no_objects_files.append(xml_file)
                continue
            
            # 각 객체 파싱
            for obj_index, obj in enumerate(objects):
                try:
                    # 클래스 이름 확인
                    name_elem = obj.find("name")
                    if name_elem is None or not name_elem.text:
                        logger.warning(f"object에 name 요소가 없음: {xml_file}, object #{obj_index+1}")
                        class_name = "unknown"  # 기본값 사용
                    else:
                        class_name = name_elem.text
                    
                    # 바운딩 박스 확인
                    bndbox = obj.find("bndbox")
                    if bndbox is None:
                        logger.warning(f"object에 bndbox 요소가 없음: {xml_file}, object #{obj_index+1}")
                        invalid_objects.append((xml_file, obj_index+1, "no_bndbox"))
                        continue
                    
                    # 바운딩 박스 좌표 파싱
                    try:
                        x_min = int(float(bndbox.find("xmin").text))
                        y_min = int(float(bndbox.find("ymin").text))
                        x_max = int(float(bndbox.find("xmax").text))
                        y_max = int(float(bndbox.find("ymax").text))
                        
                        # 바운딩 박스 유효성 확인
                        if x_min >= x_max or y_min >= y_max:
                            logger.warning(f"잘못된 바운딩 박스: {xml_file}, object #{obj_index+1}, bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                            invalid_objects.append((xml_file, obj_index+1, "invalid_bbox_values"))
                            continue
                        
                        if x_min < 0 or y_min < 0:
                            logger.warning(f"부적절한 바운딩 박스 좌표: {xml_file}, object #{obj_index+1}, bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                            # 좌표를 0으로 보정
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                    
                    except (AttributeError, ValueError) as e:
                        logger.error(f"바운딩 박스 좌표 파싱 오류: {xml_file}, object #{obj_index+1}: {e}")
                        invalid_objects.append((xml_file, obj_index+1, "bbox_parsing_error"))
                        continue
                    
                    # 애노테이션 추가
                    annotations.append({
                        "image": image_name,
                        "class": class_name,
                        "bbox": [x_min, y_min, x_max, y_max]
                    })
                    
                except Exception as e:
                    logger.error(f"object 파싱 오류: {xml_file}, object #{obj_index+1}: {e}")
                    invalid_objects.append((xml_file, obj_index+1, str(e)))
            
            processed_files += 1
            if processed_files % 100 == 0:
                logger.info(f"XML 파일 파싱 진행중: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)")
                
        except Exception as e:
            logger.error(f"XML 파일 처리 중 오류 발생: {xml_file}: {e}")
            invalid_files.append(xml_file)
    
    # 파싱 결과 요약 출력
    logger.info(f"XML 파일 파싱 완료: 총 {total_files}개 중 {processed_files}개 처리")
    logger.info(f"총 {len(annotations)}개의 애노테이션 완료")
    
    if invalid_files:
        logger.warning(f"파싱할 수 없는 XML 파일: {len(invalid_files)}개")
        logger.warning(f"[처음 10개만 표시] {invalid_files[:10]}")
    
    if no_objects_files:
        logger.warning(f"object 요소가 없는 XML 파일: {len(no_objects_files)}개")
        logger.warning(f"[처음 10개만 표시] {no_objects_files[:10]}")
    
    if invalid_objects:
        logger.warning(f"유효하지 않은 객체: {len(invalid_objects)}개")
        logger.warning(f"[처음 10개만 표시] {invalid_objects[:10]}")
    
    return annotations

"""
# 이 코드는 visualization.py로 이동하였습니다.
# 애노테이션 이미지 시각화는 visualization.py의 visualize_annotated_image 함수를 사용하세요.
"""
    