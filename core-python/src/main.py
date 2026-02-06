import cv2
import time
import os

from utils.file_utils import generar_nombre_analisis, get_input_video_path
from utils.video_utils import resize_with_padding, crear_video_writer
from utils.window_utils import crear_ventana_fija
from pose.pose_detector import PoseDetector
from pose.pose_drawer import dibujar_lado


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
GUARDAR_VIDEO = True


def main():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    input_video = get_input_video_path(BASE_DIR, 1)
    output_dir = os.path.join(BASE_DIR, "media", "output")

    cap = cv2.VideoCapture(input_video)

    crear_ventana_fija("Biomecanica Ciclista", WINDOW_WIDTH, WINDOW_HEIGHT)

    detector = PoseDetector()

    writer = None
    if GUARDAR_VIDEO:
        nombre = generar_nombre_analisis(output_dir)
        path = os.path.join(output_dir, nombre)
        writer = crear_video_writer(path, WINDOW_WIDTH, WINDOW_HEIGHT, cap.get(cv2.CAP_PROP_FPS))

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            detector.detectar_lado(landmarks)
            dibujar_lado(frame, landmarks, detector.lado)

        frame = resize_with_padding(frame, WINDOW_WIDTH, WINDOW_HEIGHT)

        fps = int(1 / (time.time() - prev_time))
        prev_time = time.time()

        cv2.putText(frame, f'FPS: {fps} | Lado: {detector.nombre_lado}',
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        if writer:
            writer.write(frame)

        cv2.imshow("Biomecanica Ciclista", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
