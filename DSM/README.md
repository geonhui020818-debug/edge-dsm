# DSM - Driver Safety Monitor

> **엣지 디바이스 기반 실시간 운전자 졸음 감지 시스템**  
> CNN + Dlib + 엣지 추론을 활용한 하이브리드 아키텍처

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [감지 기능](#감지-기능)
- [모델 구성](#모델-구성)
- [데이터셋 및 학습](#데이터셋-및-학습)
- [주요 알고리즘](#주요-알고리즘)
- [디바이스별 성능 비교](#디바이스별-성능-비교)
- [설치 및 실행](#설치-및-실행)
- [팀원](#팀원)

---

## 프로젝트 개요

카메라 영상에서 운전자의 얼굴을 실시간으로 분석하여 **눈 감김 / 하품 / 고개 숙임**을 감지하고 졸음 경보를 출력하는 시스템입니다.

Flask 웹 스트리밍을 통해 브라우저에서 실시간으로 모니터링할 수 있으며,  
**마스크 착용 여부**를 자동 감지하여 가능한 감지 항목을 동적으로 조정합니다.

### 지원 디바이스

| 디바이스 | 추론 방식 | 실행 파일 |
|----------|-----------|-----------|
| Orange Pi 5 (RK3588S NPU) | RKNN NPU 추론 | `dsm_orangepi.py` |
| Raspberry Pi 3 / 4 (NPU 없음) | TFLite CPU 추론 | `dsm_rpi.py` |

---

## 시스템 아키텍처

멀티스레드 **Producer-Consumer** 패턴으로 카메라 입력, CPU 전처리, 모델 추론을 병렬 처리합니다.

```
[Thread A] 카메라 캡처
     ↓  frame_queue
[Thread B] CPU 처리
  ├─ res10 SSD 얼굴 탐지
  ├─ dlib 68점 랜드마크 추출
  └─ 눈 / 입 / 얼굴 영역 크롭
     ↓  npu_queue
[Thread C] 모델 추론
  ├─ 눈 감김 / 하품 / 마스크 / 고개 숙임 판단
  └─ 경보 생성 및 화면 렌더링
     ↓
[Flask] 웹 스트리밍 (포트 5000)
```

> Thread C의 추론 백엔드가 디바이스에 따라 달라집니다.  
> OPi5: **RKNNLite** (NPU) / RPi: **TFLite Interpreter** (CPU)

### 파이프라인 분기 (마스크 착용 여부)

```
카메라 입력 → 얼굴 탐지 (res10 SSD)
                    ↓
             마스크 착용?
      No ────────┤├──────── Yes
      ↓                      ↓
 dlib 68점 전부            dlib 눈(36~47번)만
 눈 + 하품 + 고개 숙임     눈 + 고개 숙임
```

---

## 감지 기능

| 기능 | 경보 조건 | 마스크 착용 시 |
|------|-----------|----------------|
| **눈 감김** | 양쪽 눈 모두 2초 이상 감김 | ✅ 유지 |
| **하품** | 60초 내 3회 이상 누적 | ❌ 비활성 |
| **고개 숙임** | pitch 10도↓ 또는 Y좌표 50px↓ | ✅ 유지 |
| **마스크 감지** | 착용 여부 자동 판별 | — |

---

## 모델 구성

### 디바이스별 추론 프레임워크 및 경량화

| 구분 | Raspberry Pi | Orange Pi 5 |
|------|-------------|-------------|
| 추론 백엔드 | TFLite (CPU) | RKNN (NPU) |
| 경량화 방식 | INT8 양자화 + uint8 변환 | INT8 양자화 + rknn_toolkit optimize |
| 모델 포맷 | `.tflite` | `.rknn` |

### 1. 얼굴 인식 — `res10 SSD (OpenCV DNN)`
- 입력: 300×300, 신뢰도 0.45 이상 채택
- 마스크 착용 시에도 눈+이마 영역으로 탐지 가능
- **IoU Deadzone**: IoU > 0.85이면 노이즈로 판단하여 bbox 좌표 동결

> **핵심 아이디어** — dlib의 HOG 탐지(`get_frontal_face_detector`)는 마스크 착용 시 동작 불가.  
> res10 SSD로 bbox를 먼저 획득한 뒤 `dlib.rectangle`로 직접 주입하여 `shape_predictor`만 활용.

```python
# 기존 방식 (마스크 시 동작 불가)
faces = dlib.get_frontal_face_detector()(frame)

# 우리 방식 (res10 → dlib 주입)
bbox = face_detector.predict(frame)
rect = dlib.rectangle(x1, y1, x2, y2)
shape = predictor(frame, rect)
```

### 2. 마스크 판별 — `mask_cnn`
- 입력: 150×150 RGB, INT8
- 2클래스 분류: [마스크, 노마스크]
- **프레임 스킵(5프레임마다 1회)** → NPU 부하 80% 감소

### 3. 눈/입 상태 인지 — `eye_cnn` / `mouth_cnn`
- 눈: dlib 36~47번 랜드마크 기준 크롭 + CLAHE 전처리
- 입: dlib 48~67번 랜드마크 기준 크롭 (마스크 착용 시 비활성)
- 짝수/홀수 프레임으로 교차 처리하여 부하 분산

### 4. 고개 숙임 인지 — `FSA-Net`
- Pitch / Yaw / Roll 3축 각도 추정
- **bbox +30% 확장**: 고개 숙임 시 머리카락 가림으로 인한 오인식 방지
- **영점 보정**: 시작 후 3초간 기준 pitch 및 Y좌표 측정

---

## 데이터셋 및 학습

Kaggle 오픈 데이터셋 활용

| 항목 | 수량 | 분류 |
|------|------|------|
| 눈 깜빡임 / 졸음 | 24,000장 | 눈 뜬 상태 / 감은 상태 |
| 입 벌림 / 하품 | 4,000장 | 하품 / 정상 |
| 마스크 착용 여부 | 7,500장 | 착용 / 미착용 |

**전처리 → 학습 → 변환 파이프라인**

```
정규화 (150×150, 3채널)
  → 데이터 증강 (회전 ±10°, 좌우반전, 밝기 0.7~1.3)
  → 학습 (배치 64, 에폭 30)
  → CNN → .keras
  → RPi용:   .keras → TFLite (INT8 + uint8 변환)
  → OPi5용:  .keras → RKNN  (do_quantization INT8 + optimize)
```

---

## 주요 알고리즘

### IoU Deadzone (bbox 노이즈 차단)
매 프레임 bbox가 수 픽셀씩 떨리는 문제를 이전/현재 프레임 IoU로 필터링합니다.  
IoU > 0.85이면 AI 연산 노이즈로 판단하여 좌표를 강제 동결합니다.

### EMA 스무딩 (Head Pose)
헤드포즈 각도에 지수이동평균(alpha=0.08)을 적용하여 급격한 각도 변화를 억제합니다.

```python
pitch = (0.08 * raw_pitch) + (0.92 * prev_pitch)
```

### 하품 오감지 차단
고개가 크게 숙여진 상태(pitch 10도↓ 또는 Y좌표 50px↓)에서는 하품 감지를 일시 정지합니다.  
입이 벌어지는 각도와 하품이 겹쳐 오인식되는 문제를 방지합니다.

### 프레임 스킵 전략
| 모델 | 처리 주기 |
|------|-----------|
| 마스크 판별 | 5프레임마다 1회 |
| 눈 감김 | 짝수 프레임 |
| 하품 | 홀수 프레임 |
| 고개 숙임 | 매 프레임 |

---

## 디바이스별 성능 비교

| 디바이스 | SoC / RAM | 추론 | FPS | CPU 사용률 | CPU 온도 | 메모리 |
|----------|-----------|------|-----|-----------|---------|--------|
| Raspberry Pi 3 | 1.2GHz Quad / 1GB | TFLite | 9 FPS | 50% | 50°C | 10% |
| Raspberry Pi 4 | 1.8GHz Quad / 8GB | TFLite | 10 FPS | 98% | 60°C | 6.25% |
| **Orange Pi 5 PLUS** | **RK3588S / 16GB** | **RKNN NPU** | **45 FPS** | **35%** | **55°C** | **3.7%** |

> OPi5는 NPU 전용 추론으로 FPS 약 5배, CPU 사용률은 오히려 낮은 결과를 보입니다.

---

## 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/YOUR_ID/DSM.git
cd DSM
```

### 2. 의존성 설치

```bash
# Orange Pi 5
pip install -r requirements_orangepi.txt
# rknnlite는 별도 설치 필요: https://github.com/rockchip-linux/rknn-toolkit2

# Raspberry Pi
pip install -r requirements_rpi.txt
```

### 3. 모델 파일 배치

`models/README.md` 참고하여 모델 파일을 프로젝트 루트에 배치합니다.

### 4. 실행

```bash
# Orange Pi 5
python dsm_orangepi.py

# Raspberry Pi
python dsm_rpi.py
```

브라우저에서 `http://<device-ip>:5000` 접속

---

## 팀원

| 역할 | 이름 |
|------|------|
| 팀장 | 정지욱 |
| 팀원 | 송종선 |
| 팀원 | 이건희 |
| 팀원 | 이영근 |
| 팀원 | 정양수 |
