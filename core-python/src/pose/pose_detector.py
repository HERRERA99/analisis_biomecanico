import mediapipe as mp


class PoseDetector:

    IZQ = {
        'pts': [11, 13, 15, 23, 25, 27, 29, 31],
        'lines': [(11, 13), (13, 15), (11, 23), (23, 25), (25, 27),
                  (27, 29), (27, 31), (29, 31)]
    }

    DER = {
        'pts': [12, 14, 16, 24, 26, 28, 30, 32],
        'lines': [(12, 14), (14, 16), (12, 24), (24, 26), (26, 28),
                  (28, 30), (28, 32), (30, 32)]
    }

    def __init__(self):
        self.lado_str = None
        self.pose = mp.solutions.pose.Pose(
            model_complexity=2,
            smooth_landmarks=True
        )
        self.lado = None
        self.nombre_lado = "Detectando..."

    def process(self, frame_rgb):
        return self.pose.process(frame_rgb)

    def detectar_lado(self, landmarks):
        if self.lado:
            return

        vis_izq = sum(landmarks[i].visibility for i in self.IZQ['pts'])
        vis_der = sum(landmarks[i].visibility for i in self.DER['pts'])

        if vis_izq > vis_der:
            self.lado = self.IZQ
            self.lado_str = "left"
            self.nombre_lado = "IZQUIERDO"
        else:
            self.lado = self.DER
            self.lado_str = "right"
            self.nombre_lado = "DERECHO"
