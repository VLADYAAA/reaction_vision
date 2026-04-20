from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
import json
import math

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OUTPUT_DIR = "processed_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def get_euler_angles(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        pitch = math.atan2(R[2,1], R[2,2])
        yaw = math.atan2(-R[2,0], sy)
    else:
        pitch = math.atan2(-R[1,2], R[1,1])
        yaw = math.atan2(-R[2,0], sy)
    return np.degrees(pitch), np.degrees(yaw)

@app.post("/analyze_reaction")
async def analyze_reaction(video: UploadFile = File(...), jumps_json: str = Form(...)):
    jumps = json.loads(jumps_json)
    
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        output_facial_transformation_matrixes=True,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    output_name = f"result_{os.urandom(2).hex()}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    with FaceLandmarker.create_from_options(options) as detector:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        
        # Получаем РЕАЛЬНЫЕ параметры видео
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps < 10 or source_fps > 100: source_fps = 30
        
        # Чтобы видео не было слишком быстрым, принудительно ставим 24-30 FPS при записи
        out_v = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (w, h))

        gaze_history = []
        calibration_samples = []
        calib_yaw, calib_pitch = 0, 0
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1) # Зеркальность
            cur_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            # Если OpenCV не отдает MS корректно, считаем по индексу кадра
            if cur_ms == 0 and frame_idx > 0:
                cur_ms = (frame_idx * 1000) / source_fps

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = detector.detect_for_video(mp_img, int(cur_ms))

            if res.face_landmarks and res.facial_transformation_matrixes:
                matrix = res.facial_transformation_matrixes[0]
                h_p, h_y = get_euler_angles(matrix[:3, :3])
                
                lm = res.face_landmarks[0]
                def get_eye_angles(outer_idx, inner_idx, iris_idx):
                    iris = lm[iris_idx]
                    eye_w = abs(lm[outer_idx].x - lm[inner_idx].x)
                    center_x = (lm[outer_idx].x + lm[inner_idx].x) / 2
                    center_y = (lm[outer_idx].y + lm[inner_idx].y) / 2
                    y_ang = math.degrees(math.asin(np.clip((iris.x - center_x) / (eye_w / 2 + 1e-6), -1, 1)))
                    p_ang = math.degrees(math.asin(np.clip((iris.y - center_y) / (eye_w / 2 + 1e-6), -1, 1)))
                    return y_ang, p_ang

                ly, lp = get_eye_angles(33, 133, 468)
                ry, rp = get_eye_angles(362, 263, 473)
                
                raw_y = h_y + (ly + ry) / 2
                raw_p = h_p + (lp + rp) / 2

                # Калибровка (первые 2 секунды)
                if cur_ms < 2000:
                    calibration_samples.append((raw_y, raw_p))
                    if calibration_samples:
                        calib_yaw = np.mean([s[0] for s in calibration_samples])
                        calib_pitch = np.mean([s[1] for s in calibration_samples])
                
                final_y = raw_y - calib_yaw
                final_p = raw_p - calib_pitch
                gaze_history.append({"t": cur_ms, "y": final_y, "p": final_p})

                # --- ОТРИСОВКА ---
                # Центр между глаз (уже с учетом flip)
                mid_x = int((1.0 - (lm[468].x + lm[473].x) / 2) * w)
                mid_y = int((lm[468].y + lm[473].y) / 2 * h)
                
                # Вектор (луч)
                len_vec = 200
                vx = int(mid_x + len_vec * math.sin(math.radians(final_y)))
                vy = int(mid_y - len_vec * math.sin(math.radians(final_p)))
                
                cv2.line(frame, (mid_x, mid_y), (vx, vy), (0, 255, 0), 3)
                cv2.circle(frame, (mid_x, mid_y), 5, (255, 0, 0), -1)

                # Цели
                for j in jumps:
                    if j['t'] <= cur_ms <= j['t'] + 800:
                        jx, jy = int(j['x'] * w), int(j['y'] * h)
                        cv2.circle(frame, (jx, jy), 30, (0, 0, 255), 2)

                # Инфо на экране
                cv2.putText(frame, f"T: {int(cur_ms)}ms  Y: {final_y:.1f}", (20, 30), 1, 1.2, (255, 255, 255), 2)

            out_v.write(frame)
            frame_idx += 1
            
        cap.release()
        out_v.release()
        os.remove(tmp_path)

        # РАСЧЕТ НИСТАГМА (ДРОЖАНИЯ)
        # Считаем стандартное отклонение углов во время фазы калибровки (когда взгляд должен быть статичен)
        if len(calibration_samples) > 10:
            y_vals = [s[0] for s in calibration_samples]
            p_vals = [s[1] for s in calibration_samples]
            nystagmus_index = round(float(np.std(y_vals) + np.std(p_vals)), 3)
        else:
            nystagmus_index = 0.0

        # АНАЛИЗ РЕАКЦИИ
        results = []
        for j in jumps:
            pre_j = [f for f in gaze_history if j['t']-300 < f['t'] < j['t']]
            if not pre_j: continue
            base_y = np.mean([f['y'] for f in pre_j])
            
            for f in gaze_history:
                if j['t'] + 80 < f['t'] < j['t'] + 850:
                    if abs(f['y'] - base_y) > 2.0: # Порог саккады
                        results.append(int(f['t'] - j['t']))
                        break

    return {
        "status": "success",
        "average_ms": int(np.mean(results)) if results else 0,
        "nystagmus_index": nystagmus_index, # Степень дрожания
        "video_url": f"/get_video/{output_name}",
        "count": len(results)
    }

@app.get("/get_video/{filename}")
async def get_video(filename: str):
    return FileResponse(os.path.join(OUTPUT_DIR, filename))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)