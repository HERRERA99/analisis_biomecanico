import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose


# ===============================
# Utilidades
# ===============================

def lm_xy(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def dibujar_punto(frame, punto, color=(0, 255, 0)):
    cv2.circle(frame, punto, 6, color, -1)


# ===============================
# Ángulos
# ===============================

def obtener_angulos(frame, landmarks, lado="right"):
    h, w = frame.shape[:2]

    # -------- Selección lado --------
    if lado == "right":
        hip_idx = mp_pose.PoseLandmark.RIGHT_HIP
        knee_idx = mp_pose.PoseLandmark.RIGHT_KNEE
        ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE
        shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER
        foot_idx = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST
        elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW
        heel_idx = mp_pose.PoseLandmark.RIGHT_HEEL
    else:
        hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        knee_idx = mp_pose.PoseLandmark.LEFT_KNEE
        ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE
        shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER
        foot_idx = mp_pose.PoseLandmark.LEFT_FOOT_INDEX
        wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST
        elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW
        heel_idx = mp_pose.PoseLandmark.LEFT_HEEL

    # -------- Coordenadas píxel --------
    hip = lm_xy(landmarks[hip_idx], w, h)
    knee = lm_xy(landmarks[knee_idx], w, h)
    ankle = lm_xy(landmarks[ankle_idx], w, h)
    shoulder = lm_xy(landmarks[shoulder_idx], w, h)
    foot = lm_xy(landmarks[foot_idx], w, h)
    wrist = lm_xy(landmarks[wrist_idx], w, h)
    elbow = lm_xy(landmarks[elbow_idx], w, h)
    heel = lm_xy(landmarks[heel_idx], w, h)  # talón

    # -------- Dibujar puntos --------
    for punto in [hip, knee, ankle, shoulder, foot, wrist, elbow, heel]:
        dibujar_punto(frame, punto)

    # -------- Ángulo del pie --------
    foot_horizontal = (foot[0] + 50, foot[1])  # 50 px a la derecha

    # -------- Ángulo del tronco --------
    hip_horizontal = (hip[0] + 50, hip[1])

    # -------- Devolver grupos para ángulos --------
    return {
        "articulares": {
            "rodilla": (hip, knee, ankle),
            "tobillo": (knee, ankle, foot),
            "alcance": (wrist, shoulder, hip),
            "brazo": (wrist, elbow, shoulder),
            "hombro": (hip, shoulder, elbow),
            "pie": (heel, foot, foot_horizontal),
            "tronco": (shoulder, hip, hip_horizontal)  # añadido
        },
        "plomada": (knee, foot)
    }


def calcular_angulo(p1, vertice, p2):
    """
    Calcula el ángulo interno (el más pequeño) entre tres puntos.
    Devuelve: (valor_grados, angulo_inicio_eje_x, angulo_fin_eje_x)
    """
    p1 = np.array(p1)
    vertice = np.array(vertice)
    p2 = np.array(p2)

    # Vectores base
    v1 = p1 - vertice
    v2 = p2 - vertice

    # Ángulos absolutos respecto a la horizontal (en grados)
    ang1 = np.degrees(np.atan2(v1[1], v1[0]))
    ang2 = np.degrees(np.atan2(v2[1], v2[0]))

    diff = abs(ang1 - ang2)

    # Forzamos a que siempre tome el camino más corto (el ángulo interno)
    if diff > 180:
        diff = 360 - diff

    return diff, ang1, ang2


def angulo_tronco_horizontal(hip, shoulder):
    """
    Calcula el ángulo entre el tronco y la horizontal.
    Devuelve siempre un ángulo agudo (0-90°)
    """
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    angulo_rad = np.arctan2(abs(dy), abs(dx))  # abs() fuerza ángulo agudo
    angulo_deg = np.degrees(angulo_rad)
    return angulo_deg


def calcular_plomada_rodilla(knee, foot, lado="right"):
    """
    Devuelve:
        offset_px (positivo = adelantada, negativo = retrasada)
        linea_inicio (rodilla)
        linea_fin (proyección vertical)
    """

    knee = np.array(knee)
    foot = np.array(foot)

    offset = knee[0] - foot[0]

    # Si el ciclista mira a la izquierda invertimos signo
    if lado == "left":
        offset = -offset

    proyeccion = (foot[0], knee[1])  # vertical desde rodilla

    return offset, knee, proyeccion
