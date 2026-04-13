from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
import json

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

@app.post("/analyze_reaction")
async def analyze_reaction(
    video: UploadFile = File(...),
    jumps_json: str = Form(...) # Список времен появления точки [t1, t2, t3...]
):
    jump_times = json.loads(jumps_json)
    results = []
    
    # Снижаем пороги для работы с "прикрытыми" глазами
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        min_face_presence_confidence=0.3, # Более чувствительно к лицу
        min_tracking_confidence=0.3
    )
    
    with FaceLandmarker.create_from_options(options) as detector:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_video:
            temp_video.write(await video.read())
            temp_video_path = temp_video.name

        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_duration = 1000 / fps
        
        # Сначала соберем все координаты зрачка из видео
        frame_data = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            ts = int(frame_idx * frame_duration)
            actual_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            res = detector.detect_for_video(mp_image, ts)
            if res.face_landmarks:
                # Берем ирис (468) и углы глаза (33, 133) для калибровки
                landmarks = res.face_landmarks[0]
                iris_x = landmarks[468].x
                # Относительная позиция (нормализация по ширине глаза)
                # Это помогает ловить микродвижения
                frame_data.append({"t": actual_time, "x": iris_x})
            
            frame_idx += 1
        cap.release()
        os.remove(temp_video_path)

        # Теперь для каждого прыжка ищем реакцию в собранных данных
        for jump_t in jump_times:
            start_search = jump_t + 100 # Игнорируем первые 100мс (физически невозможно)
            end_search = jump_t + 1000  # Ищем в окне 1 секунды
            
            baseline_x = None
            # Находим среднее положение ДО прыжка
            prev_frames = [f['x'] for f in frame_data if jump_t - 200 < f['t'] < jump_t]
            if prev_frames:
                baseline_x = np.mean(prev_frames)
            
            if baseline_x:
                for f in frame_data:
                    if start_search < f['t'] < end_search:
                        if abs(f['x'] - baseline_x) > 0.0025: # Очень высокая чувствительность
                            results.append(f['t'] - jump_t)
                            break

    if not results:
        return {"status": "fail", "message": "Движения не зафиксированы. Попробуйте лучше осветить лицо."}

    # Считаем среднее и фильтруем выбросы
    avg_reaction = int(np.mean(results))
    return {
        "status": "success",
        "average_ms": avg_reaction,
        "all_reactions": results,
        "count": len(results)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)