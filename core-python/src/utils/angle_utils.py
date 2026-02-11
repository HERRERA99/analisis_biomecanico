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
        suf = "_R"
    else:
        hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        knee_idx = mp_pose.PoseLandmark.LEFT_KNEE
        ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE
        shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER
        foot_idx = mp_pose.PoseLandmark.LEFT_FOOT_INDEX
        wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST
        suf = "_L"

    # -------- Coordenadas píxel --------
    hip = lm_xy(landmarks[hip_idx], w, h)
    knee = lm_xy(landmarks[knee_idx], w, h)
    ankle = lm_xy(landmarks[ankle_idx], w, h)
    shoulder = lm_xy(landmarks[shoulder_idx], w, h)
    foot = lm_xy(landmarks[foot_idx], w, h)
    wrist = lm_xy(landmarks[wrist_idx], w, h)

    # -------- Dibujar puntos --------
    dibujar_punto(frame, hip)
    dibujar_punto(frame, knee)
    dibujar_punto(frame, ankle)
    dibujar_punto(frame, shoulder)
    dibujar_punto(frame, foot)
    dibujar_punto(frame, wrist)

    # -------- Devolver grupos para ángulos --------
    return {
        "articulares": {
            "rodilla": (hip, knee, ankle),
            "cadera": (shoulder, hip, knee),
            "tobillo": (knee, ankle, foot),
            "alcance": (wrist, shoulder, hip)
        },
        "tronco": (hip, shoulder),
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
    ang1 = np.degrees(np.arctan2(v1[1], v1[0]))
    ang2 = np.degrees(np.arctan2(v2[1], v2[0]))

    # Diferencia normalizada
    diff = (ang2 - ang1) % 360

    # Forzamos a que siempre tome el camino más corto (el ángulo interno)
    if diff > 180:
        diff = 360 - diff
        return diff, ang2, ang1  # Invertimos el orden para el dibujo

    return diff, ang1, ang2


def angulo_tronco_horizontal(hip, shoulder):
    """
    Calcula el ángulo agudo entre el tronco y la horizontal.
    Devuelve un valor entre 0 y 90 grados.
    """
    # Vector del tronco (de cadera a hombro)
    # En OpenCV, Y hombro < Y cadera si el ciclista está erguido
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]

    # Usamos arctan2 para obtener el ángulo real y luego abs() y
    # lógica de ángulo agudo
    angulo_rad = np.arctan2(abs(dy), abs(dx))
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
