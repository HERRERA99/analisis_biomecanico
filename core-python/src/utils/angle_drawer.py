import cv2
import numpy as np


def _angle_between(v):
    """Devuelve ángulo absoluto del vector en grados"""
    return np.degrees(np.arctan2(v[1], v[0]))


def dibujar_angulo(frame, centro, start, end, valor, r=40):
    """
    Dibuja el arco del ángulo y el valor numérico.
    """
    # 1. Dibujar el arco
    # Calculamos el 'span' para que OpenCV sepa cuánto recorrer desde 'start'
    span = (end - start) % 360
    if span > 180: span -= 360

    cv2.ellipse(
        frame,
        centro,
        (r, r),
        0,  # Rotación de la elipse
        int(start),  # Ángulo inicial
        int(start + span),  # Ángulo final
        (0, 255, 255),
        3  # Grosor
    )

    # 2. Dibujar el texto
    # Ubicamos el texto ligeramente desplazado del vértice para que no estorbe
    texto_pos = (centro[0] + 15, centro[1] - 15)

    cv2.putText(
        frame,
        f"{int(valor)}deg",
        texto_pos,
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

