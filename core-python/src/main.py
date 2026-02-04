import cv2
import mediapipe as mp
import os
import time
import numpy as np

# --- CONFIGURACIÓN DE USUARIO ---
GUARDAR_VIDEO = True  # Cambia a False si no quieres guardar
NOMBRE_SALIDA = "analisis_ciclista.mp4"
# --------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
video_path = os.path.join(BASE_DIR, "media", "input", "sample_1.mp4")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ No se pudo abrir el vídeo")
    exit()

# --- Configuración de ventana fija 16:9 ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
cv2.namedWindow("Biomecanica Ciclista", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("Biomecanica Ciclista", WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.setWindowProperty("Biomecanica Ciclista", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)


# --- Función para mantener proporción y rellenar bordes ---
def resize_with_padding(frame, target_width, target_height):
    h, w = frame.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))

    # Crear lienzo negro 16:9
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Centrar el frame en el lienzo
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame

    return canvas


# Configuración de guardado
if GUARDAR_VIDEO:
    output_dir = os.path.join(BASE_DIR, "media", "output")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, NOMBRE_SALIDA)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_out = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(save_path, fourcc, fps_out, (WINDOW_WIDTH, WINDOW_HEIGHT))
    print(f"💾 Grabación activada: {save_path}")

# --- MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, smooth_landmarks=True)

IZQ = {'pts': [11, 13, 15, 23, 25, 27, 29, 31],
       'lines': [(11, 13), (13, 15), (11, 23), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31)]}
DER = {'pts': [12, 14, 16, 24, 26, 28, 30, 32],
       'lines': [(12, 14), (14, 16), (12, 24), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)]}

lado_detectado = None
nombre_lado = "Detectando..."
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Procesar pose ---
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Auto-detección de lado
        if lado_detectado is None:
            vis_izq = sum([landmarks[i].visibility for i in IZQ['pts']])
            vis_der = sum([landmarks[i].visibility for i in DER['pts']])
            lado_detectado = IZQ if vis_izq > vis_der else DER
            nombre_lado = "IZQUIERDO" if lado_detectado == IZQ else "DERECHO"

        # Dibujar esqueleto lateral
        for conn in lado_detectado['lines']:
            p1, p2 = landmarks[conn[0]], landmarks[conn[1]]
            if p1.visibility > 0.5 and p2.visibility > 0.5:
                cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (0, 255, 0), 3)

        for idx in lado_detectado['pts']:
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 6, (0, 0, 255), cv2.FILLED)

    # Escalar frame manteniendo proporción y rellenando bordes
    frame = resize_with_padding(frame, WINDOW_WIDTH, WINDOW_HEIGHT)

    # Info en pantalla
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps} | Lado: {nombre_lado}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Guardar frame si está activado
    if GUARDAR_VIDEO:
        out.write(frame)

    # Mostrar frame en ventana fija
    cv2.imshow("Biomecanica Ciclista", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if GUARDAR_VIDEO:
    out.release()
cv2.destroyAllWindows()
