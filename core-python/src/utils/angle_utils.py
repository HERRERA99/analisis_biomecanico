import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose


def obtener_angulos(frame, landmarks, lado="right"):
    h, w = frame.shape[:2]

    if lado == "right":
        H = mp_pose.PoseLandmark.RIGHT_HIP
        K = mp_pose.PoseLandmark.RIGHT_KNEE
        A = mp_pose.PoseLandmark.RIGHT_ANKLE
        S = mp_pose.PoseLandmark.RIGHT_SHOULDER
        F = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    else:
        H = mp_pose.PoseLandmark.LEFT_HIP
        K = mp_pose.PoseLandmark.LEFT_KNEE
        A = mp_pose.PoseLandmark.LEFT_ANKLE
        S = mp_pose.PoseLandmark.LEFT_SHOULDER
        F = mp_pose.PoseLandmark.LEFT_FOOT_INDEX

    hip = lm_xy(landmarks[H], w, h)
    knee = lm_xy(landmarks[K], w, h)
    ankle = lm_xy(landmarks[A], w, h)
    shoulder = lm_xy(landmarks[S], w, h)
    foot = lm_xy(landmarks[F], w, h)

    return {
        "rodilla": (hip, knee, ankle),
        "cadera": (shoulder, hip, knee),
        "tobillo": (knee, ankle, foot),
        "tronco": ((shoulder[0], shoulder[1] - 100), hip, shoulder)
    }



def calcular_angulo(a, b, c):
    """
    Devuelve ángulo ABC en grados
    a, b, c -> (x, y)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle


def lm_xy(lm, w, h):
    return int(lm.x * w), int(lm.y * h)
