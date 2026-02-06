import cv2


def crear_ventana_fija(nombre, width, height):
    cv2.namedWindow(nombre, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(nombre, width, height)
    cv2.setWindowProperty(nombre, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
