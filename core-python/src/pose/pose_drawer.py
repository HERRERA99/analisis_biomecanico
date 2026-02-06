import cv2


def dibujar_lado(frame, landmarks, lado):
    h, w = frame.shape[:2]

    for conn in lado['lines']:
        p1, p2 = landmarks[conn[0]], landmarks[conn[1]]
        if p1.visibility > 0.5 and p2.visibility > 0.5:
            cv2.line(frame,
                     (int(p1.x * w), int(p1.y * h)),
                     (int(p2.x * w), int(p2.y * h)),
                     (0, 255, 0), 3)

    for idx in lado['pts']:
        lm = landmarks[idx]
        if lm.visibility > 0.5:
            cv2.circle(frame,
                       (int(lm.x * w), int(lm.y * h)),
                       6, (0, 0, 255), cv2.FILLED)
