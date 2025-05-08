# dicom_to_jpeg.py

import os
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from PIL import Image

def dicom_to_jpeg(
    dicom_path: str,
    jpg_path: str,
    window_center: float = None,
    window_width: float = None,
    resize: tuple = (512, 512)
):
    """
    Converte um arquivo DICOM em JPG, aplicando Modality LUT, VOI LUT (ou janela customizada),
    normalizando para 8-bits e redimensionando.

    Parâmetros:
    - dicom_path: caminho do .dcm de entrada
    - jpg_path: caminho do .jpg de saída
    - window_center (opcional): centro da janela (se quiser sobrescrever o VOI LUT do DICOM)
    - window_width (opcional): largura da janela
    - resize: tupla (width, height) para redimensionar a imagem salva

    Exemplo de uso:
        from dicom_to_jpeg import dicom_to_jpeg
        dicom_to_jpeg("input/12345.dcm", "output/12345.jpg")
    """
    # 1. Leitura do DICOM
    ds = pydicom.dcmread(dicom_path)

    # 2. Modality LUT: cria array em HU
    arr = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)

    # 3. VOI LUT (windowing automático) ou janela custom
    if window_center is None or window_width is None:
        img = apply_voi_lut(arr, ds)
    else:
        # janela custom: centra em `window_center`, largura `window_width`
        low = window_center - (window_width / 2)
        high = window_center + (window_width / 2)
        img = np.clip(arr, low, high)
    
    # 4. Normalização para 0–255
    img = img.astype(np.float32)
    img -= np.min(img)
    img /= np.ptp(img)
    img8 = (img * 255.0).astype(np.uint8)

    # 5. Redimensiona e salva
    pil = Image.fromarray(img8).resize(resize, Image.BILINEAR)
    pil.save(jpg_path, format="JPEG")

# Helpers para uso direto via CLI, se desejar
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Converte DICOM → JPG com LUTs e windowing")
    parser.add_argument("dicom", help="Caminho do DICOM de entrada")
    parser.add_argument("jpg", help="Caminho do JPG de saída")
    parser.add_argument("--wc", type=float, help="Window Center (opcional)")
    parser.add_argument("--ww", type=float, help="Window Width (opcional)")
    parser.add_argument("--size", type=int, nargs=2, default=(512,512),
                        help="Tamanho de saída, ex: --size 512 512")
    args = parser.parse_args()

    dicom_to_jpeg(
        args.dicom,
        args.jpg,
        window_center=args.wc,
        window_width=args.ww,
        resize=tuple(args.size)
    )
