from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
import json
import math

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация компонентов MediaPipe
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def rotation_matrix_to_euler(R):
    """Конвертация матрицы вращения в углы Pitch, Yaw, Roll"""
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(R[2,1] , R[2,2])
        yaw = math.atan2(-R[2,0], sy)
        roll = math.atan2(R[1,0], R[0,0])
    else:
        pitch = math.atan2(-R[1,2], R[1,1])
        yaw = math.atan2(-R[2,0], sy)
        roll = 0
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

@app.post("/analyze_reaction")
async def analyze_reaction(
    video: UploadFile = File(...),
    jumps_json: str = Form(...) 
):
    jump_times = json.loads(jumps_json)
    results = []
    
    # Конфигурация детектора с правильными именами атрибутов (facial)
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        output_facial_transformation_matrixes=True, # ИСПРАВЛЕНО
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    
    with FaceLandmarker.create_from_options(options) as detector:
        # Сохранение видео во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_video:
            temp_video.write(await video.read())
            temp_video_path = temp_video.name

        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_duration = 1000 / fps
        
        gaze_data = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # MediaPipe требует монотонный timestamp
            mp_timestamp = int(frame_idx * frame_duration)
            actual_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            res = detector.detect_for_video(mp_image, mp_timestamp)
            
            # Проверка наличия лица и матрицы трансформации ( facial_... )
            if res.face_landmarks and res.facial_transformation_matrixes: # ИСПРАВЛЕНО
                # 1. Поворот головы (Head Pose)
                matrix = res.facial_transformation_matrixes[0] # ИСПРАВЛЕНО
                rotation_matrix = matrix[:3, :3]
                head_pitch, head_yaw, head_roll = rotation_matrix_to_euler(rotation_matrix)
                
                # 2. Поворот зрачка относительно глаза
                landmarks = res.face_landmarks[0]
                # Используем точки левого глаза: 33 (внешний), 133 (внутренний), 468 (ирис)
                eye_outer = np.array([landmarks[33].x, landmarks[33].y])
                eye_inner = np.array([landmarks[133].x, landmarks[133].y])
                iris = np.array([landmarks[468].x, landmarks[468].y])
                
                eye_center = (eye_outer + eye_inner) / 2
                eye_width = np.linalg.norm(eye_outer - eye_inner)
                
                # Коэффициент 60.0 переводит смещение в примерные градусы
                eye_yaw_offset = ((iris[0] - eye_center[0]) / eye_width) * 60.0
                eye_pitch_offset = ((iris[1] - eye_center[1]) / eye_width) * 60.0
                
                # 3. Итоговый вектор взгляда
                gaze_data.append({
                    "t": actual_time, 
                    "yaw": head_yaw + eye_yaw_offset, 
                    "pitch": head_pitch + eye_pitch_offset
                })
            
            frame_idx += 1
            
        cap.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        # Анализ реакции на прыжки точки
        for jump_t in jump_times:
            # Ищем движение в окне от 100мс до 1000мс после появления точки
            start_search = jump_t + 100 
            end_search = jump_t + 1000  
            
            # Считаем базовое положение взгляда перед прыжком
            prev_frames = [f for f in gaze_data if jump_t - 250 < f['t'] < jump_t]
            if prev_frames:
                b_yaw = np.mean([f['yaw'] for f in prev_frames])
                b_pitch = np.mean([f['pitch'] for f in prev_frames])
                
                for f in gaze_data:
                    if start_search < f['t'] < end_search:
                        # Разница векторов в градусах
                        angular_shift = math.sqrt((f['yaw'] - b_yaw)**2 + (f['pitch'] - b_pitch)**2)
                        
                        # Если взгляд сдвинулся более чем на 2 градуса — это реакция
                        if angular_shift > 2.0:
                            results.append(f['t'] - jump_t)
                            break

    # Финальный расчет
    if not results:
        return {"status": "fail", "message": "Вектор взгляда не зафиксировал четких движений."}

    # Расчет индекса нистагма (дрожания) по первым 30 стабильным кадрам
    nystagmus = round(np.std([f['yaw'] for f in gaze_data[:30]]), 3) if len(gaze_data) > 30 else 0

    return {
        "status": "success",
        "average_ms": int(np.mean(results)),
        "nystagmus_index": nystagmus,
        "count": len(results),
        "all_results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)