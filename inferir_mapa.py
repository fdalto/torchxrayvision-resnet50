import os
import torch
import torchxrayvision as xrv
from PIL import Image
import pandas as pd
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dicom_to_jpeg import dicom_to_jpeg

# Configurações
input_dir = "input"
output_dir = "output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

os.makedirs(output_dir, exist_ok=True)

print("Carregando modelo pré-treinado do torchxrayvision...")
model = xrv.models.get_model("resnet50-res512-all")
model.eval()
model.to(device)

# Hook para capturar ativação da última camada convolucional
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Registra hook no último bloco convolucional do ResNet
model.model.layer4.register_forward_hook(get_activation("layer4"))

for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".dcm")):
        continue

    filepath = os.path.join(input_dir, filename)
    print(f"Processando: {filename}")

    if filename.lower().endswith(".dcm"):
        # Salva o JPEG com Modality LUT + VOI LUT corretamente aplicados
        jpg_name   = os.path.splitext(filename)[0] + "original.jpg"
        jpg_path   = os.path.join(output_dir, jpg_name)
        dicom_to_jpeg(
            filepath,
            jpg_path,
            # não passamos wc/ww para usar a janela padrão do DICOM
            resize=(512, 512)
        )
        # Agora recarrega o JPEG para usar como fundo do heatmap
        img_pil_vis = Image.open(jpg_path).convert("RGB")
        
        # Recarrega o DICOM para extrair o array HU real p/ inferência
        dcm = pydicom.dcmread(filepath)
        img = dcm.pixel_array.astype(np.float32)
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
        img = np.clip(img, -1024, 1024)
    else:
        # Para imagens não-DICOM, mantém o pipeline anterior
        img_pil = Image.open(filepath).convert("L").resize((512, 512))
        img = np.array(img_pil, dtype=np.float32)
        jpg_name = os.path.splitext(filename)[0] + "original.jpg"
        jpg_path = os.path.join(output_dir, jpg_name)
        img_pil.save(jpg_path, "JPEG")

    # Converter imagem para tensor
    img_resized = Image.fromarray(img).resize((512, 512))
    img_tensor = torch.tensor(np.array(img_resized)).unsqueeze(0).unsqueeze(0).float().to(device)

    # Inferência com hook ativo
    with torch.no_grad():
        preds = model(img_tensor)[0].cpu().numpy()

    # Gerar CAM com base na classe de maior score
    weights = model.model.fc.weight  # shape: [N_CLASSES, 2048]
    class_index = np.argmax(preds)
    class_weights = weights[class_index].detach().cpu().numpy()

    # Ativação da camada convolucional
    feature_map = activation["layer4"].squeeze(0).cpu().numpy()  # shape: [2048, H, W]
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)

    for i, w in enumerate(class_weights):
        cam += w * feature_map[i, :, :]

    # Normalizar e redimensionar o mapa
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    cam = Image.fromarray((cam * 255).astype(np.uint8)).resize((512, 512), resample=Image.BILINEAR)
    cam = np.array(cam)

    # Colormap
    heatmap = cm.jet(cam / 255.0)[:, :, :3]  # remove alpha
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Sobreposição com a imagem original (usa img_pil_vis para DICOM, img_pil caso contrário)
    if filename.lower().endswith(".dcm"):
        img_original = np.array(img_pil_vis.convert("RGB"))
    else:
        img_original = np.array(img_pil.convert("RGB"))


    mapa = Image.fromarray((0.5 * img_original + 0.5 * heatmap).astype(np.uint8))

    mapa_name = os.path.splitext(filename)[0] + "mapa.jpg"
    mapa_path = os.path.join(output_dir, mapa_name)
    mapa.save(mapa_path, "JPEG")

    # Salvar previsões
    df = pd.DataFrame({
        "label": model.pathologies,
        "score": preds
    })
    output_filename = os.path.splitext(filename)[0] + ".csv"
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Resultado salvo em: {output_path} e {mapa_name}")

print("Processamento finalizado.")

# Criar list.txt com arquivos .csv
print("Criando lista de pacientes para o site")
pasta_csv = './' + output_dir
arquivos_csv = [arquivo for arquivo in os.listdir(pasta_csv) if arquivo.endswith('.csv')]
caminho_lista = os.path.join(pasta_csv, 'list.txt')
with open(caminho_lista, 'w') as arquivo_lista:
    for arquivo in arquivos_csv:
        arquivo_lista.write(f"{arquivo}\n")
print(f"Arquivo list.txt criado com sucesso em {caminho_lista} contendo {len(arquivos_csv)} arquivos CSV.")
