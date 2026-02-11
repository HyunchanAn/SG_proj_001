# 고분자 물성 예측 AI 시스템

본 프로젝트는 고분자 합성 실험 데이터를 분석하고 정제하여, 새로운 실험 조건(배합비, 온도, 시간 등)에 따른 수율 및 주요 물성(점도, Tg)을 예측하는 AI 시스템입니다.

## 주요 기능

1. 데이터 전처리 및 클렌징
실험 기록 데이터에서 특수 기호와 단위를 제거하고, AI 모델이 학습 가능한 수치형 데이터셋으로 정제합니다.

2. 물성 예측 모델 학습
RandomForest 알고리즘을 활용하여 점도(R2 0.83) 및 유리전이온도(Tg, R2 0.76)를 예측하는 모델을 구축합니다.

3. 예측 시나리오 시뮬레이션
가상의 레시피를 입력하여 예상되는 물성치를 즉시 도출하는 추론 기능을 제공합니다.

## 프로젝트 구조

- data_cleaned/: 정제된 데이터셋 저장소
- models/: 학습 완료된 AI 모델 저장소
- clean_data.py: 원시 데이터 정제 및 통합 스크립트
- prepare_dataset.py: 모델 학습용 피처 추출 스크립트
- train_models_rf.py: 물성별 예측 모델 학습 스크립트
- inference.py: 사용자 입력 기반 물성 예측 시뮬레이터
- development_log.txt: 프로젝트 개발 및 수정 이력
- requirements.txt: 필수 라이브러리 의존성 명세

## 설치 및 실행 방법

1. 환경 설정
가상환경을 생성하고 필요한 패키지를 설치합니다.

python -m venv .venv
source .venv/bin/activate (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

2. 데이터 정제
원본 CSV 파일을 정제하여 데이터셋을 생성합니다.

python clean_data.py

3. 모델 학습
피처를 생성하고 예측 모델을 학습시킵니다.

python prepare_dataset.py
python train_models_rf.py

4. 물성 예측 실행
시뮬레이터를 실행하여 예측 결과를 확인합니다.

python inference.py

