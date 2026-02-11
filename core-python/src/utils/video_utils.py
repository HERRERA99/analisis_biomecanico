import cv2
import numpy as np
import os
from src.utils.angle_utils import calcular_angulo
from src.utils.angle_drawer import dibujar_angulo


def resize_with_padding(frame, target_width, target_height):
    h, w = frame.shape[:2]

    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def crear_video_writer(path, width, height, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def dibujar_angulos(frame, angulos):
    for nombre, (p1, vertice, p2) in angulos.items():
        # Obtenemos los valores corregidos
        valor, ang1, ang2 = calcular_angulo(p1, vertice, p2)

        cv2.line(frame, vertice, p1, (255, 255, 255), 1)
        cv2.line(frame, vertice, p2, (255, 255, 255), 1)

        # Dibujamos
        dibujar_angulo(frame, vertice, ang1, ang2, valor)


def dibujar_tronco_horizontal(frame, hip, shoulder, color=(0, 255, 255)):
    # 1. Calculamos el vector
    v = np.array(shoulder) - np.array(hip)

    # 2. Calculamos el ángulo agudo respecto a la horizontal
    # Usamos abs en v[1] porque en imagen Y disminuye hacia arriba
    ang = abs(np.degrees(np.arctan2(v[1], v[0])))

    # Si el ángulo supera los 90 (hacia atrás) o es negativo,
    # lo normalizamos para que siempre sea el agudo respecto a la horizontal
    if ang > 90:
        ang = 180 - ang

    # --- Dibujo ---
    # línea tronco real
    cv2.line(frame, hip, shoulder, color, 3)

    # referencia horizontal (hacia adelante según la dirección del ciclista)
    # Si el hombro está a la derecha de la cadera, dibujamos a la derecha
    direccion = 100 if shoulder[0] > hip[0] else -100
    ref = (hip[0] + direccion, hip[1])
    cv2.line(frame, hip, ref, color, 1)

    # texto
    cv2.putText(frame, f"{int(abs(ang))} deg",
                (hip[0] + 10, hip[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return ang


def dibujar_plomada(frame, knee, foot, offset, color=(0, 255, 255)):
    """
    knee  -> rodilla
    foot  -> pedal/pie
    offset -> diferencia horizontal en px
    """

    # -----------------
    # 1. Línea vertical (plomada real)
    # -----------------
    vertical_fin = (knee[0], foot[1])
    cv2.line(frame, knee, vertical_fin, color, 2)

    # -----------------
    # 2. Línea horizontal (offset visual)
    # -----------------
    cv2.line(frame, vertical_fin, foot, color, 2)

    # punto pedal
    cv2.circle(frame, foot, 5, (255, 255, 255), -1)

    # -----------------
    # 3. Texto
    # -----------------
    texto = f"{int(offset)} px"

    cv2.putText(frame, texto,
                (knee[0] + 10, knee[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2)
