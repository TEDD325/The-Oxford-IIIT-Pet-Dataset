# 데이터 시각화 기능

# imports.py에서 필요한 모듈 가져오기
from imports import *
from config import *

# 로거 가져오기
logger = get_logger()

def visualize_class_distribution(df_trainval, df_test, save_dir=None):
    """훈련/검증 및 테스트 데이터셋의 클래스 분포를 시각화합니다.
    
    Args:
        df_trainval: 훈련/검증 데이터프레임
        df_test: 테스트 데이터프레임
        save_dir: 결과물을 저장할 디렉토리 경로 (기본값: None)
    
    Returns:
        str: 저장된 파일 경로
    """
    try:
        # 결과물 저장 디렉토리 설정 (상대경로 사용)
        save_dir = create_results_directories(save_dir)
        
        # 시각화 파일명 설정 (타임스탬프 포함)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(save_dir, f'class_distribution_{timestamp}.png')
        
        # matplotlib 한글 깨짐 방지를 위한 추가 설정
        plt.figure(figsize=(12, 6))
        
        # 확실한 한글 표시를 위해 색상과 글꼴 크기 조정
        font_size = 13  # 글자 크기 조절
    
        # Train/Validation 데이터 시각화
        plt.subplot(1, 2, 1)
        ax1 = df_trainval['Species'].value_counts().plot(kind='bar', color='#2E86C1')
        plt.title('훈련/검증 데이터 클래스 분포', fontsize=font_size+1)
        plt.xlabel('Species (1=고양이, 2=개)', fontsize=font_size)
        plt.ylabel('개수', fontsize=font_size)
        
        # 값을 그래프 위에 표시
        for i, v in enumerate(df_trainval['Species'].value_counts().values):
            ax1.text(i, v + 50, str(v), ha='center', fontsize=font_size)
    
        # Test 데이터 시각화
        plt.subplot(1, 2, 2)
        ax2 = df_test['Species'].value_counts().plot(kind='bar', color='#2E86C1')
        plt.title('테스트 데이터 클래스 분포', fontsize=font_size+1)
        plt.xlabel('Species (1=고양이, 2=개)', fontsize=font_size)
        plt.ylabel('개수', fontsize=font_size)
        
        # 값을 그래프 위에 표시
        for i, v in enumerate(df_test['Species'].value_counts().values):
            ax2.text(i, v + 50, str(v), ha='center', fontsize=font_size)
        
        plt.tight_layout()
        
        # 이미지 저장
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f'클래스 분포 시각화 저장 완료: {filename}')
        
        plt.show()
        
        return filename
    
    except Exception as e:
        logger.error(f'시각화 저장 중 오류 발생: {e}')
        # 오류가 발생해도 시각화는 보여주기
        plt.tight_layout()
        plt.show()
        return None

def visualize_annotated_image(df_data, annotations, image_dir, species=None, save_dir=None, column="Image", annotation='image', bbox='bbox', class_name='class', format='jpg', color=(255, 0, 0), thickness=2, fontScale=0.5, horizontal_padding=0, vertical_padding=10, axis_off=True, img_index=None):
    """애노테이션이 포함된 이미지를 시각화합니다.
    
    Args:
        df_data: 데이터프레임
        annotations: 애노테이션 리스트
        image_dir: 이미지 디렉토리 경로
        species: 특정 종(1=고양이, 2=개)의 이미지만 출력하려면 지정 (기본값: None, 모든 종 출력)
        save_dir: 결과 저장 디렉토리 (기본값: None)
        column: 이미지 이름 컬럼 (기본값: "Image")
        annotation: 애노테이션 이미지 키 (기본값: 'image')
        bbox: 바운딩 박스 키 (기본값: 'bbox')
        class_name: 클래스 이름 키 (기본값: 'class')
        format: 이미지 형식 (기본값: 'jpg')
        color: 바운딩 박스 색상 (기본값: (255, 0, 0) - 빨간색)
        thickness: 선 두께 (기본값: 2)
        fontScale: 텍스트 크기 배율 (기본값: 0.5)
        horizontal_padding: 텍스트 가로 위치 조정 (기본값: 0)
        vertical_padding: 텍스트 세로 위치 조정 (기본값: 10)
        axis_off: 축 표시 여부 (기본값: True, 축 표시 없음)
        img_index: 이미지 인덱스 (기본값: None, 랜덤 선택)
    
    Returns:
        str: 저장된 파일 경로 또는 None
    """
    try:
        # 결과물 저장 디렉토리 설정
        if save_dir is not None:
            save_dir = create_results_directories(save_dir)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(save_dir, f'annotated_image_{timestamp}.png')
        
        # 해당 종의 이미지만 필터링
        if species is not None:
            filtered_df = df_data[df_data['Species'] == species]
            if filtered_df.empty:
                logger.warning(f"지정한 종(Species={species})의 이미지가 없습니다.")
                return None
        else:
            filtered_df = df_data
        
        # 랜덤 이미지 선택 또는 지정된 인덱스 사용
        if img_index is None:
            image_sample = filtered_df.sample(1).iloc[0]
        else:
            image_sample = filtered_df.iloc[img_index]
        
        image_name = image_sample[column]
        image_path = os.path.join(image_dir, f"{image_name}.{format}")
        
        # 이미지 읽기
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"이미지 파일 읽기 오류: {image_path} - {e}")
            return None
        
        # 해당 이미지의 어노테이션 가져오기
        image_annotations = [anno for anno in annotations if anno[annotation] == f"{image_name}.{format}"]
        
        if not image_annotations:
            logger.warning(f"이미지 {image_name}에 대한 애노테이션이 없습니다.")
        
        plt.figure(figsize=(10, 8))
        
        # 바운딩 박스 그리기
        for anno in image_annotations:
            try:
                x_min, y_min, x_max, y_max = anno[bbox]
                cv2.rectangle(
                    image, 
                    (x_min, y_min), 
                    (x_max, y_max), 
                    color, 
                    thickness)
                cv2.putText(
                    image, 
                    anno[class_name], 
                    (x_min - horizontal_padding, y_min - vertical_padding), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale, 
                    color, 
                    thickness)
            except Exception as e:
                logger.error(f"바운딩 박스 그리기 오류: {anno} - {e}")
        
        # 시각화
        plt.imshow(image)
        plt.title(f"이미지: {image_name}, 바운딩 박스: {len(image_annotations)}개", fontsize=14)
        plt.axis("off" if axis_off else "on")
        
        # 결과 저장
        if save_dir is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"애노테이션 이미지 저장 완료: {filename}")
        
        plt.show()
        
        return filename if save_dir is not None else None
    
    except Exception as e:
        logger.error(f"애노테이션 이미지 시각화 오류: {e}")
        plt.close()
        return None

def visualize_sample_images(df_trainval, image_dir, save_dir=None):
    """각 클래스에서 샘플 이미지를 출력합니다.
    
    Args:
        df_trainval: 훈련/검증 데이터프레임
        image_dir: 이미지 파일이 있는 디렉토리 경로
        save_dir: 결과물을 저장할 디렉토리 경로 (기본값: None)
    
    Returns:
        저장된 파일의 경로
    """
    try:
        # 결과물 저장 디렉토리 설정 (상대경로 사용)
        save_dir = create_results_directories(save_dir)
        
        # 시각화 파일명 설정 (타임스킬프 포함)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(save_dir, f'sample_images_{timestamp}.png')
        
        plt.figure(figsize=(10, 5))
    
        # 고양이(Species=1) 이미지 출력 - 랜덤 샘플 선택
        cat_samples = df_trainval[df_trainval['Species'] == 1]
        if not cat_samples.empty:
            cat_sample = cat_samples.sample(1).iloc[0]  # 랜덤 선택
            cat_img_path = os.path.join(image_dir, cat_sample['Image'] + '.jpg')
            cat_img = mpimg.imread(cat_img_path)
    
        plt.subplot(1, 2, 1)
        plt.imshow(cat_img)
        plt.title(f'고양이(Species=1): {cat_sample["Image"]}', fontsize=13)
        plt.axis('off')
    
        # 개(Species=2) 이미지 출력 - 랜덤 샘플 선택
        dog_samples = df_trainval[df_trainval['Species'] == 2]
        if not dog_samples.empty:
            dog_sample = dog_samples.sample(1).iloc[0]  # 랜덤 선택
            dog_img_path = os.path.join(image_dir, dog_sample['Image'] + '.jpg')
            dog_img = mpimg.imread(dog_img_path)
    
        plt.subplot(1, 2, 2)
        plt.imshow(dog_img)
        plt.title(f'개(Species=2): {dog_sample["Image"]}', fontsize=13)
        plt.axis('off')
    
        plt.tight_layout()
        
        # 이미지 저장
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f'샘플 이미지 시각화 저장 완료: {filename}')
        
        plt.show()
        
        return filename
    
    except Exception as e:
        logger.error(f'샘플 이미지 시각화 저장 중 오류 발생: {e}')
        # 오류가 발생해도 시각화는 보여주기
        plt.tight_layout()
        plt.show()
        return None
