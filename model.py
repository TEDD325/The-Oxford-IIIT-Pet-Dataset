from imports import *
from config import *
import logging
import os
import datetime
import time
import torch
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

logger = logging.getLogger(__name__)

def save_model(model, filepath):
    """
    학습된 모델을 파일로 저장합니다.
    
    Args:
        model: 저장할 PyTorch 모델
        filepath: 저장할 파일 경로
    """
    # 저장 디렉토리 확인 및 생성
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"생성된 디렉토리: {directory}")
    
    # 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': model.head.classification_head.num_classes,
        'saved_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, filepath)
    
    logger.info(f"모델 저장 완료: {filepath}")


def load_model(filepath, device):
    """
    저장된 모델을 불러오는 함수
    
    Args:
        filepath: 모델 파일 경로
        device: 모델이 실행될 장치
        
    Returns:
        PyTorch 모델: 불러온 모델
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")
    
    # 모델 파일 불러오기
    checkpoint = torch.load(filepath, map_location=device)
    
    # 기본 모델 생성
    model = torchvision.models.detection.ssd300_vgg16(
        weights=SSD300_VGG16_Weights.COCO_V1,
        weights_backbone=VGG16_Weights.IMAGENET1K_V1,
    ).to(device)
    
    # 클래스 수 설정
    num_classes = checkpoint.get('num_classes', 91)  # 기본값은 COCO 클래스 수(91)
    model.head.classification_head.num_classes = num_classes
    
    # 저장된 가중치 불러오기
    model.load_state_dict(checkpoint['model_state_dict'])
    
    saved_at = checkpoint.get('saved_at', '알 수 없음')
    logger.info(f"모델 불러오기 성공: {filepath} (저장 시간: {saved_at})")
    
    return model


def initialize_model(classes, device, model_path=None):
    """
    SSD 모델을 초기화하고 클래스 수에 맞게 설정
    - Single Shot MultiBox Detector(SSD) 객체 검출 알고리즘의 구현체
        - Faster R-CNN: 영역 제안(Region Proposal)과 분류를 별도로 수행
        - SSD는 단일 네트워크에서 객체 감지와 분류를 동시에 수행
        - 특징 추출을 위해 VGG16 네트워크를 백본으로 사용
        - ImageNet으로 사전 학습된 가중치를 활용
        - 다양한 크기의 객체를 검출하기 위해 다중 스케일 특징 맵 사용
        - 300x300 픽셀 크기의 입력 이미지 사용
        - 다양한 종형비와 크기의 앵커 박스 사용 -> 객체 검출 성능 향상
        - 정확도와 속도의 균형이 좋아 실시간 객체 검출에 자주 사용됨
    
    Args:
        classes: 분류할 클래스 리스트
        device: 모델이 실행될 장치(GPU/CPU)
        model_path: 불러올 모델 파일 경로 (선택사항, None일 경우 처음부터 초기화)
        
    Returns:
        초기화된 SSD 모델
    """
    # 저장된 모델 불러오기 시도
    if model_path is not None:
        try:
            logger.info(f"저장된 모델 불러오기 시도: {model_path}")
            model = load_model(model_path, device)
            
            # 클래스 수가 다를 경우 헤드를 재초기화
            if model.head.classification_head.num_classes != len(classes):
                logger.warning(f"모델 클래스 수 불일치: 저장된 모델({model.head.classification_head.num_classes}개) vs 현재 클래스({len(classes)}개)")
                model.head.classification_head.num_classes = len(classes)
                logger.info(f"모델 헤드 재초기화 완료: {len(classes)}개 클래스로 설정됨")
            
            return model
            
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"모델 불러오기 오류: {e}")
            logger.info("새 모델을 초기화합니다.")
    
    # 새 모델 초기화
    logger.info("SSD300_VGG16 모델 초기화 중...")
    # SSD 모델 불러오기
    model = torchvision.models.detection.ssd300_vgg16(
        weights=SSD300_VGG16_Weights.DEFAULT,
        weights_backbone=VGG16_Weights.IMAGENET1K_V1,
        num_classes=91
        ).to(device)
    """
    weights=SSD300_VGG16_Weights.DEFAULT
        - COCO 데이터셋(80개 클래스)에 대해 사전 학습된 가중치
        - 객체 검출에 특화되어 좋은 성능 보장함
        - 그 이외의 가능한 가중치
            - None: 사전 학습된 가중치 사용하지 않음. 램덤 초기화 수행
            - SSD300_VGG16_Weights.COCO_V1: 이전 버전의 COCO 가중치 사용
            - SSD300_VGG16_Weights.DEFAULT: 최신 버전의 기본 가중치 사용
            - 직접 학습한 가중치 파일 경로로도 지정 가능
        Q. 사전 학습된 가중치 사용하는 것의 이점은?
        A. 
            - 처음부터 학습하지 않고도 좋은 성능을 얻을 수 있다.
            - 전이 학습(transfer learning)을 통해 적은 양의 데이터로도 특정 작업에 모델을 효과적으로 적응시킬 수 있다.
    """
    
    # 클래스 개수에 맞게 출력 레이어 수정
    model.head.classification_head.num_classes = len(classes)
    
    # COCO 모델의 state_dict 추출
    coco_sd = model.state_dict()

    # 3-class 모델 생성
    model = ssd300_vgg16(
        weights=None,
        weights_backbone=VGG16_Weights.IMAGENET1K_V1,
        num_classes=len(classes)
    ).to(device)

    # COCO 가중치 중, shape이 일치하는 파라미터만 덮어쓰기
    own_sd = model.state_dict()
    for k, v in coco_sd.items():
        if k in own_sd and own_sd[k].shape == v.shape:
            own_sd[k] = v
    model.load_state_dict(own_sd)
    

    logger.info(f"모델 초기화 완료: {len(classes)}개 클래스로 설정됨")
    return model

def calculate_iou(box, boxes):
    """
    하나의 박스와 다수의 박스 사이의 IoU(Intersection over Union)를 계산
    - IoU(Intersection over Union)는 예측된 바운딩 박스와 실제 바운딩 박스 간의 겹침 정도를 측정하는 핵심 지표
    - IoU 값이 특정 임계값(일반적으로 0.5) 이상일 때 해당 예측을 맞은 것(True Positive)으로 판단
    - 객체 검출 알고리즘은 하나의 객체에 대해 여러 개의 겹치는 바운딩 박스를 예측할 수 있는데, IoU를 사용하여 중복된 박스를 제거할 수 있음
    - AP(Average Precision)와 mAP(mean Average Precision) 계산에 IoU가 필수적으로 사용됨
    
    Args:
        box: [x_min, y_min, x_max, y_max] 형태의 하나의 박스
        boxes: [N, 4] 형태의 다수의 박스, 각 행은 [x_min, y_min, x_max, y_max]
        
    Returns:
        ndarray: 박스와 각 박스 사이의 IoU 값의 배열
    """
    # 각 박스 쌍 사이의 교집합 영역 계산
    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])

    # 교집합 영역의 면적 계산 (비정상 교집합은 0으로 처리)
    intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    
    # 각 박스의 면적 계산
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 합집합(union) 면적 = 박스A 면적 + 박스B 면적 - 교집합 면적
    union = box_area + boxes_area - intersection

    # IoU = 교집합 / 합집합
    iou = intersection / union
    return iou


def calculate_class_ap(predictions, ground_truths, class_idx, iou_threshold=0.5):
    """
    특정 클래스에 대한 Average Precision(AP)를 계산
    - True Positive(TP): IoU가 임계값 이상이고 아직 매칭되지 않은 정답 박스와 매칭된 예측
    - False Positive(FP): IoU가 임계값 미만이거나 이미 매칭된 정답 박스와 매칭된 예측
    
    
    Args:
        predictions: 모델의 예측 결과를 담고 있는 리스트
        ground_truths: 실제 정답을 담고 있는 리스트
        class_idx: 평가할 클래스 인덱스
        iou_threshold: IoU(Intersection over Union) 임계값, 기본값 0.5
        
    Returns:
        float: 해당 클래스의 Average Precision(AP) 값
    """
    scores = []  # 확률 값 저장 리스트
    labels = []  # 정답 여부 저장 리스트 (1: 정답, 0: 오답)
    all_gt_boxes = 0  # 전체 정답 박스 개수

    # 각 이미지별 예측과 정답을 비교
    for pred, gt in zip(predictions, ground_truths):
        # 예측 박스, 레이블, 확률 값 가져오기
        pred_boxes = np.array(pred["boxes"])
        pred_labels = np.array(pred["labels"])
        pred_scores = np.array(pred.get("scores", np.ones(len(pred_boxes))))

        # 정답 박스와 레이블 가져오기
        gt_boxes = np.array(gt["boxes"])
        gt_labels = np.array(gt["labels"])

        # 현재 클래스에 해당하는 박스만 필터링
        pred_boxes = pred_boxes[pred_labels == class_idx]
        pred_scores = pred_scores[pred_labels == class_idx]
        gt_boxes = gt_boxes[gt_labels == class_idx]

        # 전체 정답 박스 개수 누적
        all_gt_boxes += len(gt_boxes)

        # 예측과 정답 둘 다 없으면 스킵
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue

        # 이미 매칭된 정답 박스 추적
        matched = np.zeros(len(gt_boxes), dtype=bool)

        # 각 예측 박스에 대해 처리
        for box, score in zip(pred_boxes, pred_scores):
            # 정답 박스가 없는 경우, 해당 예측은 무조건 오답
            if len(gt_boxes) == 0:
                scores.append(score)
                labels.append(0)  # False positive
                continue

            # 예측 박스와 모든 정답 박스 사이의 IoU 계산
            ious = calculate_iou(box, gt_boxes)
            max_idx = np.argmax(ious)  # 가장 높은 IoU를 가진 정답 박스 인덱스
            max_iou = ious[max_idx]  # 가장 높은 IoU 값

            # IoU가 임계값 이상이고, 아직 매칭되지 않은 정답 박스인 경우
            if max_iou >= iou_threshold and not matched[max_idx]:
                labels.append(1)  # True positive
                matched[max_idx] = True  # 정답 박스 매칭 표시
            else:
                labels.append(0)  # False positive

            scores.append(score)

    # 해당 클래스의 예측이 없는 경우 AP = 0
    if len(labels) == 0:
        return 0.0

    # scikit-learn의 average_precision_score 함수를 사용해 AP 계산
    return average_precision_score(labels, scores)


def evaluate_model(predictions, ground_truths, classes):
    """
    데이터셋에 대한 모델의 성능을 평가합니다.
    
    Args:
        predictions: 모델의 예측 결과 리스트
        ground_truths: 실제 정답 리스트
        classes: 클래스 이름 리스트 (background를 포함)
        
    Returns:
        float: Mean Average Precision (mAP) 값
    """
    logger.info(f"모델 평가 시작: {len(predictions)}개 이미지, {len(classes)-1}개 클래스")
    class_aps = []  # 각 클래스별 AP 값 저장 리스트

    # background(0)를 제외한 모든 클래스에 대해 처리
    for class_idx, class_name in enumerate(classes[1:], start=1):
        logger.debug(f"클래스 '{class_name}' 평가 중...")
        
        # calculate_class_ap 함수를 사용하여 클래스별 AP 계산
        ap = calculate_class_ap(predictions, ground_truths, class_idx)
        class_aps.append(ap)
        logger.debug(f"클래스 '{class_name}' AP: {ap:.4f}")
    
    # 모든 클래스의 AP 평균값 계산 (mAP)
    mAP = np.mean(class_aps)
    logger.info(f"평가 완료 - Mean Average Precision (mAP): {mAP:.4f}")
    
    # 각 클래스별 AP 값 출력
    for i, (class_name, ap) in enumerate(zip(classes[1:], class_aps)):
        logger.info(f"  - {class_name}: {ap:.4f}")
        
    return mAP


def evaluate_model_detailed(predictions, ground_truths, classes, iou_threshold=0.5):
    """
    데이터셋에 대한 모델의 성능을 상세히 평가하는 함수입니다.
    AP 및 정밀도/재현도 그래프를 생성할 수 있습니다.
    
    Args:
        predictions: 모델의 예측 결과 리스트
        ground_truths: 실제 정답 리스트
        classes: 클래스 이름 리스트 (background를 포함)
        iou_threshold: IoU 임계값, 기본값 0.5
        
    Returns:
        dict: 다양한 평가 지표를 포함한 사전
    """
    class_aps = []
    class_precisions = {}
    class_recalls = {}
    
    # 각 클래스별로 처리 (background 제외)
    for class_idx, class_name in enumerate(classes[1:], start=1):
        true_positives = []
        scores = []
        num_ground_truths = 0

        # 각 이미지에 대해 평가 진행
        for pred, gt in zip(predictions, ground_truths):
            # 현재 클래스에 해당하는 박스만 필터링
            pred_boxes = pred["boxes"][pred["labels"] == class_idx].cpu().numpy() if len(pred["boxes"]) > 0 else []
            pred_scores = pred["scores"][pred["labels"] == class_idx].cpu().numpy() if len(pred["scores"]) > 0 else []
            gt_boxes = gt["boxes"][gt["labels"] == class_idx].cpu().numpy() if len(gt["boxes"]) > 0 else []

            num_ground_truths += len(gt_boxes)

            # 예측 또는 정답 박스가 없는 경우 스킵
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue

            # 이미 매칭된 정답 박스 표시
            matched = np.zeros(len(gt_boxes), dtype=bool)
            
            # 각 예측 박스에 대해 처리
            for box, score in zip(pred_boxes, pred_scores):
                ious = calculate_iou(box, gt_boxes)
                max_iou_idx = np.argmax(ious) if len(ious) > 0 else -1
                max_iou = ious[max_iou_idx] if max_iou_idx >= 0 else 0

                # IoU가 임계값 이상이고 아직 매칭되지 않은 정답 박스인 경우
                if max_iou >= iou_threshold and not matched[max_iou_idx]:
                    true_positives.append(1)  # 참 양성 (True Positive)
                    matched[max_iou_idx] = True  # 해당 정답 박스는 이미 매칭됨
                else:
                    true_positives.append(0)  # 거짓 양성 (False Positive)

                scores.append(score)

        # 해당 클래스에 대한 예측이 없는 경우, AP = 0
        if len(scores) == 0:
            class_aps.append(0)
            class_precisions[class_name] = [0]
            class_recalls[class_name] = [0]
            continue

        # 확률값 기준 내림차순 정렬 (높은 확률부터)
        sorted_indices = np.argsort(-np.array(scores))
        true_positives = np.array(true_positives)[sorted_indices]
        scores = np.array(scores)[sorted_indices]

        # 누적 TP 계산
        cum_true_positives = np.cumsum(true_positives)
        
        # 정밀도(Precision)와 재현도(Recall) 계산
        precision = cum_true_positives / (np.arange(len(true_positives)) + 1)
        recall = cum_true_positives / max(num_ground_truths, 1)  # 0으로 나누는 경우 방지

        # 해당 클래스의 AP 계산
        ap = average_precision_score(true_positives, scores) if len(scores) > 0 else 0
        class_aps.append(ap)
        
        # 클래스별 정밀도/재현도 저장
        class_precisions[class_name] = precision
        class_recalls[class_name] = recall

    # 총합 결과 계산
    mAP = np.mean(class_aps)
    
    # 결과 사전 구성
    results = {
        "mAP": mAP,
        "class_APs": {name: ap for name, ap in zip(classes[1:], class_aps)},
        "precisions": class_precisions,
        "recalls": class_recalls
    }
    
    return results

def train_model(model, train_loader, val_loader, classes, device, 
              num_epochs=5, lr=0.001, momentum=0.9, weight_decay=0.0005, 
              step_size=3, gamma=0.1, save_dir='models', model_name='ssd_model'):
    """
    모델 학습 및 평가를 위한 함수
    
    Args:
        model: 학습할 SSD 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        classes: 클래스 이름 리스트
        device: 모델이 실행될 장치 (CPU/GPU)
        num_epochs: 총 학습 에폭 수, 기본값 5
        lr: 학습률, 기본값 0.001
        momentum: SGD 옵티마이저의 모멘텀 계수, 기본값 0.9
        weight_decay: 가중치 감소 계수, 기본값 0.0005
        step_size: 학습률 스케줄러의 스텝 크기, 기본값 3
        gamma: 학습률 감소 비율, 기본값 0.1
        save_dir: 모델 저장 디렉토리, 기본값 'models'
        model_name: 모델 파일 이름 접두사, 기본값 'ssd_model'
        
    Returns:
        dict: 학습 결과 및 평가 지표를 포함한 사전
    """
    logger.info(f"모델 학습 시작: {num_epochs} 에폭, 학습률 {lr}")
    
    # 모델 저장 디렉토리 확인
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info(f"모델 저장 디렉토리 생성: {save_dir}")
    
    # 학습 속도 개선을 위한 옵티마이저 변경 (SGD -> Adam)
    # Adam은 일반적으로 학습 속도가 더 빠름
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 더 효율적인 학습률 스케줄러 사용 - ReduceLROnPlateau
    # 검증 손실이 개선되지 않을 때 학습률을 감소시킴
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=gamma, patience=2, verbose=True
    )
    
    # 학습 결과 저장을 위한 변수
    training_results = {
        "train_losses": [],
        "val_maps": [],
        "best_map": 0.0,
        "best_epoch": 0,
        "best_model_path": None,
        "last_model_path": None
    }
    
    # MacOS의 MPS 가속을 위한 gradients 누적 단계 설정
    # 작은 배치를 누적하여 효율성 개선
    gradient_accumulation_steps = 4
    
    # Training + Validation Loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs} 시작")
        
        # Training Phase
        model.train()
        total_train_loss = 0
        batch_count = 0
        accumulated_loss = 0
        
        # tqdm을 사용하여 학습 진행도 표시
        for idx, (images, targets) in enumerate(tqdm(train_loader, desc="Training")):
            batch_count += 1
            
            # 데이터를 지정된 장치로 이동 (MPS 메모리 사용량 최적화)
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            # Forward pass - 모델 출력과 손실 계산
            # torch.no_op() 대신 nullcontext 사용
            with (torch.cuda.amp.autocast() if device.type == 'cuda' else nullcontext()):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                # Gradient Accumulation - 기본 batch size가 작은 경우 여러 배치를 누적
                loss = losses / gradient_accumulation_steps
            
            # Backward pass 및 가중치 업데이트
            loss.backward()  # 그래디언트 계산
            accumulated_loss += losses.item()
            
            # gradient_accumulation_steps마다 업데이트 수행
            if (idx + 1) % gradient_accumulation_steps == 0 or (idx + 1) == len(train_loader):
                optimizer.step()  # 가중치 업데이트
                optimizer.zero_grad()  # 그래디언트 초기화
                total_train_loss += accumulated_loss
                accumulated_loss = 0
        
        # 학습 에폭의 평균 손실 계산
        avg_train_loss = total_train_loss / len(train_loader)
        training_results["train_losses"].append(avg_train_loss)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation Phase - 검증 데이터로 모델 평가 (최적화)
        model.eval()  # 모델을 평가 모드로 전환
        all_predictions = []
        all_ground_truths = []
        
        # 가중치 업데이트 없이 추론만 수행 (메모리 최적화)
        with torch.no_grad():
            # 더 큰 배치 크기로 평가하여 속도 향상
            for images, targets in tqdm(val_loader, desc="Validation"):
                # 비동기 데이터 전송으로 성능 향상
                images = [img.to(device, non_blocking=True) for img in images]
                
                # 모델 추론 수행 (메모리 사용량 최적화)
                # torch.no_op() 대신 nullcontext 사용
                with (torch.cuda.amp.autocast() if device.type == 'cuda' else nullcontext()):
                    predictions = model(images)  # 추론 모드에서는 targets 없이 호출
                
                # CPU로 이동하여 메모리 최적화
                processed_predictions = [
                    {
                        "boxes": p["boxes"].detach().cpu(),
                        "labels": p["labels"].detach().cpu(),
                        "scores": p["scores"].detach().cpu()
                    } for p in predictions
                ]
                
                # 추론 결과와 Ground Truth 저장
                all_predictions.extend(processed_predictions)
                all_ground_truths.extend(targets)
        
        # mAP 계산을 통한 성능 평가
        mAP = evaluate_model(all_predictions, all_ground_truths, classes)
        training_results["val_maps"].append(mAP)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Validation mAP: {mAP:.4f}")
        
        # 현재 에폭 모델 저장
        current_model_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")
        save_model(model, current_model_path)
        
        # 최고 성능 기록 및 모델 저장
        if mAP > training_results["best_map"]:
            training_results["best_map"] = mAP
            training_results["best_epoch"] = epoch + 1
            
            # 최고 모델 저장
            best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")
            save_model(model, best_model_path)
            training_results["best_model_path"] = best_model_path
            
            logger.info(f"새로운 최고 mAP: {mAP:.4f} (Epoch {epoch + 1}) - 모델 저장: {best_model_path}")
        
        # 학습률 스케줄러 업데이트
        lr_scheduler.step()
        logger.info("-" * 60)
    
    # 마지막 모델 저장
    last_model_path = os.path.join(save_dir, f"{model_name}_final.pth")
    save_model(model, last_model_path)
    training_results["last_model_path"] = last_model_path
    
    logger.info(f"학습 완료! 최고 mAP: {training_results['best_map']:.4f} (Epoch {training_results['best_epoch']})")
    logger.info(f"최고 모델 경로: {training_results['best_model_path']}")
    logger.info(f"최종 모델 경로: {training_results['last_model_path']}")
    
    return training_results


def test_model(model, test_loader, classes, device):
    """
    테스트 데이터셋에 대한 모델 평가 함수
    
    Args:
        model: 평가할 SSD 모델
        test_loader: 테스트 데이터 로더
        classes: 클래스 이름 리스트
        device: 모델이 실행될 장치
        
    Returns:
        dict: 테스트 결과 및 추론 예측값을 포함한 사전
    """
    logger.info("테스트 데이터셋 평가 시작")
    model.eval()
    
    all_predictions = []
    all_images = []  # 이미지 저장 (추후 시각화에 활용 가능)
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Testing"):
            all_images.extend(images)  # 원본 이미지 저장
            images = [img.to(device) for img in images]
            predictions = model(images)
            all_predictions.extend(predictions)
    
    logger.info(f"테스트 완료: {len(all_predictions)}개 이미지 추론")
    
    # 추론 결과 반환
    return {
        "predictions": all_predictions,
        "images": all_images
    }