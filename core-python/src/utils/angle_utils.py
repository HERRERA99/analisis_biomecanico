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
        suf = "_R"
    else:
        hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        knee_idx = mp_pose.PoseLandmark.LEFT_KNEE
        ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE
        shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER
        foot_idx = mp_pose.PoseLandmark.LEFT_FOOT_INDEX
        suf = "_L"

    # -------- Coordenadas píxel --------
    hip = lm_xy(landmarks[hip_idx], w, h)
    knee = lm_xy(landmarks[knee_idx], w, h)
    ankle = lm_xy(landmarks[ankle_idx], w, h)
    shoulder = lm_xy(landmarks[shoulder_idx], w, h)
    foot = lm_xy(landmarks[foot_idx], w, h)

    # -------- Dibujar puntos --------
    dibujar_punto(frame, hip)
    dibujar_punto(frame, knee)
    dibujar_punto(frame, ankle)
    dibujar_punto(frame, shoulder)
    dibujar_punto(frame, foot)

    # -------- Devolver grupos para ángulos --------
    return {
        "rodilla": (hip, knee, ankle),
        "cadera": (shoulder, hip, knee),
        "tobillo": (knee, ankle, foot),
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
    ang1 = np.degrees(np.arctan2(v1[1], v1[0]))
    ang2 = np.degrees(np.arctan2(v2[1], v2[0]))

    # Diferencia normalizada
    diff = (ang2 - ang1) % 360

    # Forzamos a que siempre tome el camino más corto (el ángulo interno)
    if diff > 180:
        diff = 360 - diff
        return diff, ang2, ang1  # Invertimos el orden para el dibujo

    return diff, ang1, ang2