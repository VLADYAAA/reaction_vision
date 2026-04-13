import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import mediapipe as mp
import numpy as np
import av
import time
import cv2

# --- ПРАВИЛЬНЫЙ ИМПОРТ (на основе твоего теста) ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Инициализация детектора (нужен файл face_landmarker.task в папке!)
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO
)
detector = FaceLandmarker.create_from_options(options)

class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_x = None
        self.detection_time = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Конвертация кадра
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Детекция (используем текущее время в мс)
        result = detector.detect_for_video(mp_image, int(time.time() * 1000))

        if result.face_landmarks:
            # Точка 468 — центр левого зрачка
            iris = result.face_landmarks[0][468]
            
            if self.last_x is not None:
                dx = abs(iris.x - self.last_x)
                if dx > 0.01: # Резкое движение глаза
                    self.detection_time = time.time()
            
            self.last_x = iris.x
            
            # Визуализация для отладки
            h, w, _ = img.shape
            cv2.circle(img, (int(iris.x * w), int(iris.y * h)), 4, (0, 255, 0), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ---
st.set_page_config(layout="wide")
st.title("Sober or Drunk Analysis Lab")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Видеопоток")
    ctx = webrtc_streamer(
        key="eye-test",
        video_processor_factory=EyeProcessor,
        mode=WebRtcMode.SENDRECV
    )

with col2:
    st.subheader("Тест реакции")
    
    # Кнопка и летящая точка через HTML/JS
    js_component = """
    <div id="box" style="width:100%; height:300px; background:#222; position:relative; border-radius:10px; overflow:hidden;">
        <div id="dot" style="width:20px; height:20px; background:red; border-radius:50%; position:absolute; display:none;"></div>
        <button id="start" style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); padding:10px;">Начать</button>
    </div>
    <script>
        const dot = document.getElementById('dot');
        const btn = document.getElementById('start');
        btn.onclick = () => {
            btn.style.display = 'none';
            setTimeout(() => {
                dot.style.left = Math.random()*90 + '%';
                dot.style.top = Math.random()*90 + '%';
                dot.style.display = 'block';
                // Отправляем сигнал в Python о появлении точки
                window.parent.postMessage({type: 'STIMULUS', time: Date.now()}, '*');
            }, 1000 + Math.random()*2000);
        };
    </script>
    """
    components.html(js_component, height=350)
    
    if st.button("Показать результат"):
        if ctx.video_processor and ctx.video_processor.detection_time:
             st.write(f"Последнее зафиксированное движение: {ctx.video_processor.detection_time}")
        else:
             st.info("Движение еще не зафиксировано")