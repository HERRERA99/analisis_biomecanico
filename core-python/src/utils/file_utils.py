import os


def generar_nombre_analisis(ruta_carpeta):
    # Definimos las extensiones de video que queremos considerar
    extensiones_video = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')

    try:
        # Listamos los archivos en la carpeta
        archivos = os.listdir(ruta_carpeta)

        # Filtramos y contamos solo los que son videos
        num_videos = sum(1 for archivo in archivos if archivo.lower().endswith(extensiones_video))

        # Retornamos el string formateado con el conteo
        return f"analisis_ciclista_{num_videos}.mp4"

    except FileNotFoundError:
        return "Error: La carpeta no existe."


def get_input_video_path(base_dir, numero, extension=".mp4"):
    """
    1 -> media/input/sample_1.mp4
    """
    filename = f"sample_{numero}{extension}"
    return os.path.join(base_dir, "media", "input", filename)
