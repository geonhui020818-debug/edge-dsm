# models/

이 디렉토리에 아래 모델 파일들을 위치시켜야 합니다.  
용량 문제로 깃허브에는 포함되지 않습니다.

## Orange Pi 5용 (.rknn)

| 파일명 | 용도 |
|--------|------|
| `fsanet.rknn` | 고개 숙임 (Head Pose) |
| `eye_new.rknn` | 눈 감김 감지 |
| `mouth_new.rknn` | 하품 감지 |
| `mask_new.rknn` | 마스크 착용 여부 |

## Raspberry Pi용 (.tflite)

| 파일명 | 용도 |
|--------|------|
| `fsanet.tflite` | 고개 숙임 (Head Pose) |
| `eye_cnn_int8.tflite` | 눈 감김 감지 |
| `mouth_cnn_int8.tflite` | 하품 감지 |
| `mask_cnn_int8.tflite` | 마스크 착용 여부 |

## 공통 파일

| 파일명 | 용도 |
|--------|------|
| `deploy.prototxt` | res10 SSD 얼굴 탐지 구조 |
| `res10_300x300_ssd_iter_140000.caffemodel` | res10 SSD 가중치 |
| `shape_predictor_68_face_landmarks.dat` | dlib 68점 랜드마크 |
