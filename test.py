import torch

# CUDA 사용 가능 여부 확인
print("CUDA 사용 가능 여부:", torch.cuda.is_available())

# 사용 가능한 GPU 수와 이름 출력
if torch.cuda.is_available():
    print(f"사용 가능한 GPU 수: {torch.cuda.device_count()}")
    print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
    
    # 기본 장치 설정
    device = torch.device('cuda')
    
    # 텐서 생성 및 GPU 연산
    x = torch.rand(100, device=device)
    print(x)
    
    # GPU 메모리 사용량 확인
    print(f"할당된 메모리: {torch.cuda.memory_allocated()/1e6:.2f} MB")
    print(f"캐시된 메모리: {torch.cuda.memory_reserved()/1e6:.2f} MB")
else:
    print("CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
    device = torch.device('cpu')
    x = torch.rand(100, device=device)
    print(x)