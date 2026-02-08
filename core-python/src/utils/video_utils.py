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
    for puntos in angulos.values():
        a, b, c = puntos
        valor = calcular_angulo(a, b, c)
        dibujar_angulo(frame, a, b, c, valor)

