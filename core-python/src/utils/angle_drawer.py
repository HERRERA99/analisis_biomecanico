import cv2
import numpy as np


def _angle_between(v):
    """Devuelve ángulo absoluto del vector en grados"""
    return np.degrees(np.arctan2(v[1], v[0]))


def dibujar_angulo(frame, a, b, c, valor, radio=40, color=(0, 255, 0)):
    """
    Dibuja:
      - líneas BA y BC
      - arco del ángulo
      - texto con grados

    a, b, c -> puntos (x,y) donde B es vértice
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    ang1 = _angle_between(ba)
    ang2 = _angle_between(bc)

    start = min(ang1, ang2)
    end = max(ang1, ang2)

    # arco (quesito)
    cv2.ellipse(
        frame,
        tuple(b),
        (radio, radio),
        0,
        start,
        end,
        color,
        2
    )

    # texto dentro del arco
    mid = np.radians((start + end) / 2)
    text_pos = (
        int(b[0] + radio * 0.7 * np.cos(mid)),
        int(b[1] + radio * 0.7 * np.sin(mid))
    )

    cv2.putText(
        frame,
        f"{int(valor)}°",
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )
