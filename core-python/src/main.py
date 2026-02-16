import cv2
import time
import os

from src.utils.angle_utils import obtener_angulos, calcular_plomada_rodilla, calcular_angulo, mp_pose
from src.utils.recorder_utils import DataRecorder
from utils.file_utils import generar_nombre_analisis, get_input_video_path
from utils.video_utils import (
    resize_with_padding, crear_video_writer, dibujar_angulos, dibujar_plomada, crear_layout_dashboard,
    dibujar_info_dashboard
)
from utils.window_utils import crear_ventana_fija
from pose.pose_detector import PoseDetector
from pose.pose_drawer import dibujar_lado

# Configuramos el ancho total sumando el panel lateral (1280 + 350)
ANCHO_PANEL = 350
WINDOW_WIDTH = 1280 + ANCHO_PANEL
WINDOW_HEIGHT = 720
GUARDAR_VIDEO = False


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    input_video = get_input_video_path(BASE_DIR, 4)
    output_dir = os.path.join(BASE_DIR, "media", "output")

    cap = cv2.VideoCapture(input_video)
    crear_ventana_fija("Biomecanica Ciclista", WINDOW_WIDTH, WINDOW_HEIGHT)

    detector = PoseDetector()
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    if GUARDAR_VIDEO:
        nombre = generar_nombre_analisis(output_dir)
        path = os.path.join(output_dir, nombre)
        writer = crear_video_writer(path, WINDOW_WIDTH, WINDOW_HEIGHT, fps_video)

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Procesamiento Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        torso_ang = 0
        offset_kops = 0
        angulos_val = {"rodilla": 0, "cadera": 0, "tobillo": 0, "alcance": 0}

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            detector.detectar_lado(landmarks)

            # Dibujo sobre el ciclista (lado izquierdo)
            dibujar_lado(frame, landmarks, detector.lado)
            datos = obtener_angulos(frame, landmarks, detector.lado_str)
            dibujar_angulos(frame, datos["articulares"])

            knee, foot = datos["plomada"]
            offset_kops, k, proj = calcular_plomada_rodilla(knee, foot, detector.lado_str)

            color_plomada = (0, 255, 0) if abs(offset_kops) < 10 else (0, 0, 255)
            dibujar_plomada(frame, k, foot, offset_kops, color_plomada)

            for nombre, (p1, v, p2) in datos["articulares"].items():
                valor, _, _ = calcular_angulo(p1, v, p2)
                angulos_val[nombre] = valor

        # 2. CREACIÓN DEL DASHBOARD
        # Primero redimensionamos el video original
        frame_resized = resize_with_padding(frame, 1280, 720)

        # Creamos el layout extendido
        layout, x_panel = crear_layout_dashboard(frame_resized, ancho_panel=ANCHO_PANEL)

        # Calculamos FPS de procesamiento
        fps_proc = int(1 / (time.time() - prev_time))
        prev_time = time.time()

        # Dibujamos la info en el panel derecho
        dibujar_info_dashboard(
            layout, x_panel, angulos_val, offset_kops,
            torso_ang, detector.nombre_lado, fps_proc
        )

        # 3. Salida
        if writer:
            writer.write(layout)

        cv2.imshow("Biomecanica Ciclista", layout)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if writer: writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
