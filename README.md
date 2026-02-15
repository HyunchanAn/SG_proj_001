# 고분자 물성 예측 AI 시스템 (SG_proj_001)

본 프로젝트는 고분자 합성 및 도포 실험 데이터를 분석하여, 새로운 실험 조건(배합비, 온도, 시간, 기재 등)에 따른 주요 물성을 예측하고 목표 물성을 위한 최적 배합비를 제안하는 AI 시스템입니다.

## 배포 주소
- URL: https://sg-proj-001.streamlit.app/

## 주요 기능

1. **데이터 전처리 및 클렌징**
   - 실험 기록 데이터에서 특수 기호와 단위를 제거하고, AI 모델 학습이 가능한 수치형 데이터셋으로 정제합니다.
   
2. **멀티 타겟 물성 예측 모델**
   - **합성 모델**: RandomForest 기반 점도(R2 0.75), 유리전이온도(Tg, R2 0.88) 예측
   - **도포 모델**: 원단(기재) 종류 및 경화제 함량에 따른 점착력(R2 0.74) 예측
   
3. **지능형 역설계 (Inverse Design)**
   - 원하는 목표 물성($T_g$ 등)을 입력하면, AI 최적화 엔진이 최적의 모노머 배합비를 역산하여 추천합니다.
   
4. **시각화 대시보드**
   - 예측 결과값과 함께 입력 데이터 분포 및 배합비 구성을 차트로 실시간 시각화합니다.

## 프로젝트 구조

- `data_cleaned/`: 정제된 데이터셋 저장소
- `models/`: 학습 완료된 AI 모델 및 피처 리스트 저장소
- `scripts/`: 데이터 정제, 피처 추출, 모델 학습, 역설계 엔진용 스크립트 모음
- `app.py`: Streamlit 기반 웹 시뮬레이터 메인 파일
- `development_log.txt`: 프로젝트 개발 및 수정 상세 이력
- `requirements.txt`: 필수 라이브러리 의존성 명세

## 설치 및 실행 방법

1. **환경 설정**
   가상환경을 생성하고 필요한 패키지를 설치합니다.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
   pip install -r requirements.txt
   ```

2. **전체 파이프라인 실행 (데이터 정제 -> 모델 학습)**
   ```bash
   python scripts/clean_data.py
   python scripts/prepare_dataset.py
   python scripts/prepare_coating_dataset.py
   python scripts/train_models_rf.py
   python scripts/train_coating_models.py
   ```

3. **시뮬레이터 실행**
   ```bash
   streamlit run app.py
   ```

## 업데이트 사항 (2026-02-15)
- UI/UX 전면 개편 (그리드 레이아웃, 모노머 범주화, Expander 적용)
- 역설계 결과의 합성 탭 자동 연동 기능 (배합비 및 공정 조건 풀 동기화)
- 데이터 아티팩트 필터링 로직 강화 및 모델 재학습 완료
