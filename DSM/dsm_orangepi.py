from rknnlite.api import RKNNLite
import cv2
import dlib
import numpy as np
import time
import threading
import psutil
import os
import queue
from flask import Flask, Response, render_template_string

# ── 1. 설정값 ─────────────────────────────────────────────
PROTOTXT_PATH     = "deploy.prototxt"
CAFFEMODEL_PATH   = "res10_300x300_ssd_iter_140000.caffemodel"

FSANET_MODEL_PATH = "fsanet.rknn" 
EYE_MODEL_PATH    = "eye_new.rknn"
MOUTH_MODEL_PATH  = "mouth_new.rknn"
MASK_MODEL_PATH   = "mask_new.rknn"

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
IMG_SIZE       = (150, 150)
EYE_IMG_SIZE   = (150, 150)

CAM_INDEX = 0
CAM_W     = 640
CAM_H     = 480

EYE_THRESH        = 0.7  
EYE_DROP_SEC      = 2.0   

MOUTH_THRESH      = 0.55 
YAWN_WINDOW       = 60.0   
YAWN_COUNT_THRESH = 3      

CALIBRATION_SEC   = 3.0    
PITCH_DROP_DELTA  = 10.0   
Y_DROP_PIXELS     = 50.0   
HEAD_DROP_SEC     = 2.0    

MASK_THRESH  = 0.5

LEFT_EYE_IDX  = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))
MOUTH_IDX     = list(range(48, 68))

app          = Flask(__name__)
output_frame = None
lock         = threading.Lock()

frame_queue = queue.Queue(maxsize=2)
npu_queue   = queue.Queue(maxsize=2)

def put_latest(q, item):
    if q.full():
        try: q.get_nowait()
        except queue.Empty: pass
    q.put(item)

# ── 1.5. 하드웨어 온도 측정 유틸 ─────────────────────────
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read().strip()) / 1000.0
        return temp
    except Exception:
        return 0.0

# ── 2. 모델 클래스 모음 ──────────────────────────────
class Res10SSDFaceDetector:
    def __init__(self, prototxt, caffemodel, conf_thresh=0.45):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.conf_thresh = conf_thresh
    def predict(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        best_box = None
        max_conf = 0.0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_thresh:
                if confidence > max_conf:
                    max_conf = confidence
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    best_box = box.astype("int")
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            return (x1, y1, x2, y2)
        return None
    def release(self): pass

class RKNNHeadPose:
    def __init__(self, model_path):
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    def predict(self, face_rgb_64x64):
        inp = np.ascontiguousarray(face_rgb_64x64)[np.newaxis, ...]
        outputs = self.rknn.inference(inputs=[inp])
        if outputs is None or len(outputs) == 0: return 0.0, 0.0, 0.0
        res = outputs[0][0]
        return float(res[0]), float(res[1]), float(res[2])
    def release(self): self.rknn.release()

class RKNNModelEye:
    def __init__(self, model_path):
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    def predict(self, img_float32):
        inp = (img_float32 * 255).astype(np.float32)[np.newaxis]
        outputs = self.rknn.inference(inputs=[inp])
        if outputs is None: return 0.0
        return float(outputs[0][0][0])
    def release(self): self.rknn.release()

class RKNNModelSoftmax:
    def __init__(self, model_path):
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    def predict(self, img_float32):
        inp = (img_float32 * 255).astype(np.float32)[np.newaxis]
        outputs = self.rknn.inference(inputs=[inp])
        if outputs is None: return [0.0, 0.0]
        out = outputs[0][0]
        out_exp = np.exp(out - out.max())
        return out_exp / out_exp.sum()
    def release(self): self.rknn.release()

# ── 3. 유틸 함수 ─────────────────────
def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype=np.float32)
    for i in range(68): coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def preprocess_eye(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 150: enhanced = cv2.convertScaleAbs(enhanced, alpha=0.8, beta=0)
    return enhanced

def crop_by_indices(frame, pts, indices, padding=1.5, is_eye=False):
    region = pts[indices]
    x1, y1 = region.min(axis=0).astype(int)
    x2, y2 = region.max(axis=0).astype(int)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = int((x2 - x1) * padding)
    H, W = frame.shape[:2]
    x1, x2 = max(0, cx - half), min(W, cx + half)
    y1, y2 = max(0, cy - half), min(H, cy + half)
    if x2 <= x1 or y2 <= y1: return None
    crop = frame[y1:y2, x1:x2]
    if is_eye:
        crop = preprocess_eye(crop)
        return cv2.cvtColor(cv2.resize(crop, EYE_IMG_SIZE), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        return cv2.cvtColor(cv2.resize(crop, IMG_SIZE), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def crop_face(frame, rect):
    H, W = frame.shape[:2]
    x1, y1 = max(0, rect.left() - 20), max(0, rect.top() - 20)
    x2, y2 = min(W, rect.right() + 20), min(H, rect.bottom() + 20)
    if x2 <= x1 or y2 <= y1: return None
    crop = cv2.resize(frame[y1:y2, x1:x2], IMG_SIZE)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def draw_landmarks(frame, pts, indices, is_alert, label):
    color = (0, 0, 255) if is_alert else (0, 255, 0)
    region_pts = pts[indices].astype(np.int32)
    for (x, y) in region_pts: cv2.circle(frame, (x, y), 2, color, -1)
    cv2.polylines(frame, [region_pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(frame, label, (int(region_pts[:, 0].min()), int(region_pts[:, 1].min()) - 6), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

# ═════════════════════════════════════════════════════════════════════════════
# 4. 비동기 멀티스레딩 아키텍처 (Producer-Consumer)
# ═════════════════════════════════════════════════════════════════════════════

def camera_thread():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    
    frame_cnt = 0
    print("[Thread A] Camera Thread Started")
    
    while True:
        ret, cap_frame = cap.read()
        if not ret: break
        cap_frame = cv2.flip(cap_frame, 1)
        now = time.time()
        
        put_latest(frame_queue, (cap_frame, frame_cnt, now))
        frame_cnt += 1

    cap.release()

def cpu_thread():
    face_detector = Res10SSDFaceDetector(PROTOTXT_PATH, CAFFEMODEL_PATH)
    predictor     = dlib.shape_predictor(PREDICTOR_PATH)
    
    prev_box = None
    missing_frames = 0
    MAX_MISSING_FRAMES = 15  
    SKIP_FRAMES = 3 
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    
    print("💻 [Thread B] CPU Worker Thread Started")
    
    while True:
        frame, frame_cnt, frame_time = frame_queue.get()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = frame.shape[:2]
        
        crops = {} 
        pts = None
        face_found_now = False

        if frame_cnt % SKIP_FRAMES == 0 or prev_box is None:
            face_box = face_detector.predict(frame)
            if face_box is not None:
                missing_frames = 0  
                raw_x1, raw_y1, raw_x2, raw_y2 = face_box
                face_found_now = True
            else:
                missing_frames += SKIP_FRAMES 
                if prev_box is not None:
                    raw_x1, raw_y1, raw_x2, raw_y2 = prev_box
                    face_found_now = True
        else:
            if prev_box is not None:
                raw_x1, raw_y1, raw_x2, raw_y2 = prev_box
                face_found_now = True
        
        if face_found_now and missing_frames < MAX_MISSING_FRAMES:
            box_w = raw_x2 - raw_x1
            box_h = raw_y2 - raw_y1
            p_x1, p_x2 = max(0, raw_x1 - int(box_w * 0.05)), min(W, raw_x2 + int(box_w * 0.05))
            p_y1, p_y2 = max(0, raw_y1), min(H, raw_y2 + int(box_h * 0.05))

            if prev_box is None:
                x1, y1, x2, y2 = p_x1, p_y1, p_x2, p_y2
            else:
                ix1 = max(p_x1, prev_box[0])
                iy1 = max(p_y1, prev_box[1])
                ix2 = min(p_x2, prev_box[2])
                iy2 = min(p_y2, prev_box[3])
                
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                box1_area = (p_x2 - p_x1) * (p_y2 - p_y1)
                box2_area = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
                iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)

                if iou > 0.85:
                    x1, y1, x2, y2 = prev_box  
                else:
                    alpha = 0.2 
                    x1 = int(alpha * p_x1 + (1 - alpha) * prev_box[0])
                    y1 = int(alpha * p_y1 + (1 - alpha) * prev_box[1])
                    x2 = int(alpha * p_x2 + (1 - alpha) * prev_box[2])
                    y2 = int(alpha * p_y2 + (1 - alpha) * prev_box[3])
            
            prev_box = (x1, y1, x2, y2)
            dlib_rect = dlib.rectangle(x1, y1, x2, y2)

            gray_enhanced = clahe.apply(gray)
            shape = predictor(gray_enhanced, dlib_rect)
            pts = shape_to_np(shape)

            if frame_cnt % 5 == 0:
                crops['mask'] = crop_face(frame, dlib_rect)

            fw, fh = x2 - x1, y2 - y1
            hx1, hy1 = max(0, x1 - int(fw * 0.3)), max(0, y1 - int(fh * 0.3))
            hx2, hy2 = min(W, x2 + int(fw * 0.3)), min(H, y2 + int(fh * 0.3))
            face_crop_pose = frame[hy1:hy2, hx1:hx2]
            
            if face_crop_pose.shape[0] > 10 and face_crop_pose.shape[1] > 10:
                face_64 = cv2.resize(face_crop_pose, (64, 64))
                crops['pose'] = cv2.cvtColor(face_64, cv2.COLOR_BGR2RGB)

            if frame_cnt % 2 == 0:
                crops['left_eye'] = crop_by_indices(frame, pts, LEFT_EYE_IDX, 1.5, is_eye=True)
                crops['right_eye'] = crop_by_indices(frame, pts, RIGHT_EYE_IDX, 1.5, is_eye=True)
            else:
                crops['mouth'] = crop_by_indices(frame, pts, MOUTH_IDX, 1.2)
        else:
            prev_box = None

        put_latest(npu_queue, (frame, frame_cnt, frame_time, prev_box, pts, crops))

def npu_thread():
    global output_frame

    head_pose   = RKNNHeadPose(FSANET_MODEL_PATH)
    eye_model   = RKNNModelEye(EYE_MODEL_PATH)
    mouth_model = RKNNModelSoftmax(MOUTH_MODEL_PATH)
    mask_model  = RKNNModelSoftmax(MASK_MODEL_PATH)

    prev_yaw = prev_pitch = prev_roll = None
    angle_alpha = 0.08  
    
    is_calibrated = False
    calibration_start = None
    pitch_history = []
    baseline_pitch = 0.0
    
    y_history = []
    baseline_y = 0.0
    
    last_lp = last_rp = mouth_prob = mask_prob = 0.0
    last_is_yawn = wearing_mask = False
    
    eye_closed_start = head_drop_start = None
    yawn_timestamps = []
    yawn_warning_end_time = 0.0
    
    prev_time = time.time()
    last_log_time = time.time() 
    avg_fps = 0.0 
    
    print("🧠 [Thread C] NPU Worker Thread Started (Tuned Yawn Mute)")

    while True:
        frame, frame_cnt, frame_time, face_box, pts, crops = npu_queue.get()
        now = time.time()
        
        inst_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        
        if avg_fps == 0.0:
            avg_fps = inst_fps
        else:
            avg_fps = (0.1 * inst_fps) + (0.9 * avg_fps)
            
        H, W = frame.shape[:2]
        active_warnings = []
        head_drowsy = False 
        is_head_dropped_severely = False 

        if now - last_log_time >= 2.0:
            cpu_usage = psutil.cpu_percent()
            mem_usage = psutil.virtual_memory().percent
            temp = get_cpu_temp()
            temp_str = f"{temp:.1f}°C" + (" (🔥과열)" if temp > 75.0 else "")
            print(f"[모니터링] Avg FPS: {avg_fps:>4.1f} | CPU: {cpu_usage:>4.1f}% | RAM: {mem_usage:>4.1f}% | Temp: {temp_str}")
            last_log_time = now

        if face_box is not None and pts is not None:
            x1, y1, x2, y2 = face_box
            current_cy = (y1 + y2) / 2.0

            if 'mask' in crops and crops['mask'] is not None:
                mask_prob = float(mask_model.predict(crops['mask'])[0])
                wearing_mask = mask_prob > MASK_THRESH

            if 'pose' in crops:
                raw_yaw, raw_pitch, raw_roll = head_pose.predict(crops['pose'])
                
                if prev_pitch is None:
                    yaw, pitch, roll = raw_yaw, raw_pitch, raw_roll
                else:
                    yaw = (angle_alpha * raw_yaw) + ((1 - angle_alpha) * prev_yaw)
                    pitch = (angle_alpha * raw_pitch) + ((1 - angle_alpha) * prev_pitch)
                    roll = (angle_alpha * raw_roll) + ((1 - angle_alpha) * prev_roll)
                
                prev_yaw, prev_pitch, prev_roll = yaw, pitch, roll
                
                if not is_calibrated:
                    if calibration_start is None:
                        calibration_start = frame_time
                        
                    pitch_history.append(pitch)
                    y_history.append(current_cy)
                    
                    if frame_time - calibration_start >= CALIBRATION_SEC:
                        baseline_pitch = sum(pitch_history) / len(pitch_history)
                        baseline_y = sum(y_history) / len(y_history) 
                        is_calibrated = True
                        print(f"🎯 [영점 조절] Pitch: {baseline_pitch:.1f}도 / Y좌표: {baseline_y:.1f}px")
                    
                    active_warnings.append("CALIBRATING PITCH & Y-AXIS... LOOK FORWARD")
                    head_drop_start = None
                else:
                    is_pitch_dropped = pitch <= (baseline_pitch - PITCH_DROP_DELTA)
                    is_y_dropped = current_cy >= (baseline_y + Y_DROP_PIXELS)
                    
                    # 🚀 [버그 수정] 애매한 7도 기준을 지우고, 확실하게 고개가 꺾였거나(10도) Y좌표가 떨어진(50px) 경우만 하품을 차단!
                    is_head_dropped_severely = is_pitch_dropped or is_y_dropped
                    
                    if is_pitch_dropped or is_y_dropped:
                        if head_drop_start is None: head_drop_start = frame_time
                        if is_y_dropped and not is_pitch_dropped:
                            cv2.putText(frame, "CATCH BY Y-AXIS DROP", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        head_drop_start = None

                    head_drowsy = (head_drop_start is not None) and (now - head_drop_start >= HEAD_DROP_SEC)
                    if head_drowsy: active_warnings.append("!! HEAD DROP WARNING !!")

            if frame_cnt % 2 == 0:
                lp = eye_model.predict(crops['left_eye']) if 'left_eye' in crops and crops['left_eye'] is not None else 0.0
                rp = eye_model.predict(crops['right_eye']) if 'right_eye' in crops and crops['right_eye'] is not None else 0.0
                both_closed = (lp > EYE_THRESH) and (rp > EYE_THRESH)
                
                if both_closed:
                    if eye_closed_start is None: eye_closed_start = frame_time
                else:
                    eye_closed_start = None
                last_lp, last_rp = lp, rp

            eye_drowsy = (eye_closed_start is not None) and (now - eye_closed_start >= EYE_DROP_SEC)
            if eye_drowsy: active_warnings.append("!! EYE CLOSED WARNING !!")

            if frame_cnt % 2 != 0:
                if wearing_mask:
                    last_is_yawn = False
                    yawn_timestamps.clear()
                    yawn_count = 0
                    yawn_warning_end_time = 0.0 
                elif is_head_dropped_severely:
                    last_is_yawn = False
                else:
                    if 'mouth' in crops and crops['mouth'] is not None:
                        mouth_prob = float(mouth_model.predict(crops['mouth'])[1])
                        is_yawn = (mouth_prob > MOUTH_THRESH)
                        if is_yawn and not last_is_yawn: yawn_timestamps.append(frame_time)
                        last_is_yawn = is_yawn
                    else:
                        last_is_yawn = False

            if not wearing_mask:
                yawn_timestamps = [t for t in yawn_timestamps if now - t < YAWN_WINDOW]
                yawn_count = len(yawn_timestamps)
                
                if yawn_count >= YAWN_COUNT_THRESH:
                    yawn_warning_end_time = now + 3.0  
                    yawn_timestamps.clear()            
                    yawn_count = 0                     
                
                if now < yawn_warning_end_time:
                    active_warnings.append("!! YAWN WARNING !!")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            fw, fh = x2 - x1, y2 - y1
            hx1, hy1 = max(0, x1 - int(fw * 0.3)), max(0, y1 - int(fh * 0.3))
            hx2, hy2 = min(W, x2 + int(fw * 0.3)), min(H, y2 + int(fh * 0.3))
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 1)

            draw_landmarks(frame, pts, LEFT_EYE_IDX, last_lp > EYE_THRESH, "CLOSED" if last_lp > EYE_THRESH else "OPEN")
            draw_landmarks(frame, pts, RIGHT_EYE_IDX, last_rp > EYE_THRESH, "CLOSED" if last_rp > EYE_THRESH else "OPEN")
            
            if wearing_mask: 
                draw_landmarks(frame, pts, MOUTH_IDX, False, "MASK ON")
            elif is_head_dropped_severely:
                draw_landmarks(frame, pts, MOUTH_IDX, False, "PAUSED (Head Drop)")
            else: 
                draw_landmarks(frame, pts, MOUTH_IDX, last_is_yawn, "YAWN" if last_is_yawn else "NO YAWN")
            
            if prev_pitch is not None:
                if is_calibrated:
                    pose_text = f"P:{pitch:.1f} Y-Drop:{int(current_cy - baseline_y)}px"
                else:
                    pose_text = f"P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}"
                    
                cv2.putText(frame, pose_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if not head_drowsy else (0,0,255), 2)

        else:
            if not is_calibrated:
                calibration_start = None
                pitch_history = []
                y_history = []
                
            prev_yaw = prev_pitch = prev_roll = None
            eye_closed_start = head_drop_start = None
            
            yawn_timestamps.clear()
            last_lp = last_rp = mouth_prob = mask_prob = 0.0
            last_is_yawn = wearing_mask = False
            yawn_count = 0
            yawn_warning_end_time = 0.0
            cv2.putText(frame, "No Face Detected", (W//2 - 120, H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80,80,255), 2)

        if active_warnings:
            overlay = frame.copy()
            box_height = 40 + len(active_warnings) * 40
            cv2.rectangle(overlay, (0, H//2 - 40), (W, H//2 - 40 + box_height), (0,0,200), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            for i, msg in enumerate(active_warnings):
                sz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
                cv2.putText(frame, msg, ((W-sz[0])//2, H//2 + i*40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2)

        if wearing_mask:
            yawn_ui_text = "Yawn: OFF (Mask ON)"
        elif is_head_dropped_severely:
            yawn_ui_text = f"Yawn: PAUSED (Head Drop)"
        else:
            yawn_ui_text = f"Yawn: {yawn_count}/{YAWN_COUNT_THRESH} (60s)"
        
        infos = [f"FPS: {avg_fps:.1f} (Optimized)", 
                 f"Eye Drop: {'ON' if eye_closed_start else 'OFF'}", 
                 yawn_ui_text, 
                 f"Head Drop: {'ON' if head_drop_start else 'OFF'}"]
        for i, text in enumerate(infos):
            cv2.putText(frame, text, (10, 22 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,255,200), 1)

        with lock: output_frame = frame.copy()

@app.route('/video')
def video_feed():
    def generate():
        global output_frame
        while True:
            with lock:
                if output_frame is None:
                    time.sleep(0.01)
                    continue
                ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret: continue
                frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html><head><style>body { background:#111; color:white; text-align:center; }</style></head>
    <body><h1>졸음 감지 시스템 (멀티스레드 최적화 버전)</h1><img src="/video" style="border:2px solid #4af; max-width: 100%;"></body></html>
    ''')

if __name__ == "__main__":
    t1 = threading.Thread(target=camera_thread)
    t2 = threading.Thread(target=cpu_thread)
    t3 = threading.Thread(target=npu_thread)
    
    t1.daemon = t2.daemon = t3.daemon = True
    
    t1.start()
    t2.start()
    t3.start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True)
