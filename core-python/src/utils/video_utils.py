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
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def crear_video_writer(path, width, height, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def dibujar_angulos(frame, angulos):
    for nombre, (p1, vertice, p2) in angulos.items():
        valor, ang1, ang2 = calcular_angulo(p1, vertice, p2)
        dibujar_angulo(frame, vertice, ang1, ang2, valor)


def dibujar_plomada(frame, knee, foot, offset, color=(0, 255, 255)):
    vertical_fin = (knee[0], foot[1])
    cv2.line(frame, knee, vertical_fin, color, 2)
    cv2.line(frame, vertical_fin, foot, color, 2)
    cv2.circle(frame, foot, 5, (255, 255, 255), -1)


def crear_layout_dashboard(frame, ancho_panel=350):
    """Crea un espacio extra a la derecha del frame original"""
    h, w = frame.shape[:2]
    panel = np.zeros((h, ancho_panel, 3), dtype=np.uint8)
    # Línea divisoria
    cv2.line(panel, (0, 0), (0, h), (50, 50, 50), 2)
    layout = np.hstack((frame, panel))
    return layout, w


def dibujar_info_dashboard(layout, x_start, datos_angulos, kops, torso_ang, lado_nombre, fps):
    """Dibuja dinámicamente solo los ángulos presentes y la plomada"""
    f = cv2.FONT_HERSHEY_SIMPLEX
    y = 40
    c_tit = (0, 255, 255)  # Cian/Amarillo
    c_txt = (255, 255, 255)  # Blanco

    # -------- Cabecera --------
    cv2.putText(layout, "SISTEMA BIOMECANICO", (x_start + 20, y), f, 0.9, c_tit, 2)
    y += 40
    cv2.putText(layout, f"Lado: {lado_nombre.upper()}", (x_start + 20, y), f, 0.6, c_txt, 1)
    y += 25
    cv2.putText(layout, f"FPS: {fps}", (x_start + 20, y), f, 0.5, (150, 150, 150), 1)

    # -------- Medidas dinámicas --------
    y += 50
    cv2.putText(layout, "MEDIDAS ACTIVAS:", (x_start + 20, y), f, 0.6, c_tit, 2)
    y += 40

    # Diccionario de traducción y rangos
    config = {
        "rodilla": {"label": "Rodilla", "rango": (35, 155)},
        "tobillo": {"label": "Tobillo", "rango": (70, 110)},
        "alcance": {"label": "Alcance", "rango": (75, 90)},
        "brazo": {"label": "Brazo", "rango": (120, 175)},
        "hombro": {"label": "Hombro", "rango": (70, 90)},
        "pie": {"label": "Inclin. Pie", "rango": (0, 30)},
        "tronco": {"label": "Tronco", "rango": (35, 55)}
    }

    for clave, valor in datos_angulos.items():
        if valor == 0: continue  # Ignorar si no se ha detectado/calculado

        color_val = c_txt
        label = config.get(clave, {}).get("label", clave.capitalize())
        rango = config.get(clave, {}).get("rango")

        # Alerta visual si sale de rango
        if rango:
            if not (rango[0] <= valor <= rango[1]):
                color_val = (0, 0, 255)  # Rojo

        txt = f"{label}: {int(valor)}°"
        cv2.putText(layout, txt, (x_start + 20, y), f, 0.6, color_val, 1)
        y += 35

    # -------- KOPS (Plomada) --------
    y += 20
    cv2.line(layout, (x_start + 20, y), (x_start + 300, y), (50, 50, 50), 1)
    y += 40

    # El color de KOPS ya viene validado o lo validamos aquí
    color_kops = (0, 255, 0) if abs(kops) < 15 else (0, 0, 255)
    cv2.putText(layout, "PLOMADA (KOPS):", (x_start + 20, y), f, 0.6, c_tit, 2)
    y += 35
    cv2.putText(layout, f"Offset: {int(kops)} px", (x_start + 20, y), f, 0.7, color_kops, 1)

