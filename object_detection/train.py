# 학습 파라미터 설정
import torch
from ultralytics import YOLO
import os
import json



training_args = {
    'data': 'data_pr.yaml',  # 데이터셋 설정 파일 경로
    'epochs': 300,  # 전체 학습 에포크
    'batch': -1,    # 배치 크기
    'imgsz': 512,   # 입력 이미지 크기
    'patience': 100,  # Early stopping patience
    'device': 0 if torch.cuda.is_available() else 'cpu',  # GPU 사용 여부
    'workers': 4,   # 데이터 로더 워커 수
    'name': 'test_pr_final',  # 실험 이름 (학습 ID 사용)
    'exist_ok': True,  # 기존 실험 결과 덮어쓰기 허용
    'pretrained': True,  # 사전 학습된 가중치 사용
    'optimizer': 'AdamW',  # 옵티마이저 선택
    'lr0': 0.001,  # 초기 학습률
    'hsv_h': .2,
    'hsv_s': .2,
    'hsv_v': .2,
    'weight_decay': 0.01,  # 가중치 감쇠
    'warmup_epochs': 5,
    'cos_lr': True,  # Cosine 학습률 스케줄러 사용
}

if __name__=='__main__':
    # freeze_support()...?
    model= YOLO('./yolo11n.pt')

    model.train(**training_args)


