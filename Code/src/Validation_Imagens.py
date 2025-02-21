# Validação do Modelo VGG16 em Novas Imagens com Limiar Dinâmico

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing import image

# Caminho das imagens
novas_imagens_path = "dataset/novas_imagens"

# Carregar o modelo treinado
#model_path = "models/vgg16_pneumonia_finetuned_v2.h5"
model_path = "models/mobilenetv2_pneumonia_finetuned_v2.h5"
model = tf.keras.models.load_model(model_path)

# Parâmetros
IMG_SIZE = (224, 224)

# Função para processar imagem
def processar_imagem(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalização
    return img_array

# Ajuste Dinâmico do Limiar
def definir_limiar(preds):
    media = np.mean(preds)
    if media > 0.7:
        return 0.6  # Se o modelo estiver com muita certeza em pneumonia, ajustamos
    elif media < 0.4:
        return 0.5  # Se houver mais NORMAL, mantemos padrão
    else:
        return 0.55  # Caso intermediário

# Listar imagens nas subpastas NORMAL e PNEUMONIA
categorias = ["NORMAL", "PNEUMONIA"]

# Armazenar previsões para ajuste de limiar
todas_preds = []

# Processar imagens e armazenar previsões
for categoria in categorias:
    pasta_categoria = os.path.join(novas_imagens_path, categoria)
    if os.path.exists(pasta_categoria):
        imagens = [f for f in os.listdir(pasta_categoria) if f.endswith(('png', 'jpg', 'jpeg'))]
        for img_nome in imagens:
            img_path = os.path.join(pasta_categoria, img_nome)
            img_array = processar_imagem(img_path)
            pred = model.predict(img_array)[0][0]
            todas_preds.append(pred)

# Definir novo limiar com base na distribuição das previsões
novo_limiar = definir_limiar(todas_preds)
print(f"📌 Novo limiar definido dinamicamente: {novo_limiar:.2f}")

# Visualizar previsões com o novo limiar
plt.figure(figsize=(10, 5))
i = 0
for categoria in categorias:
    pasta_categoria = os.path.join(novas_imagens_path, categoria)
    if os.path.exists(pasta_categoria):
        imagens = [f for f in os.listdir(pasta_categoria) if f.endswith(('png', 'jpg', 'jpeg'))]
        for img_nome in imagens[:5]:
            img_path = os.path.join(pasta_categoria, img_nome)
            img_array = processar_imagem(img_path)
            pred = model.predict(img_array)[0][0]
            predicao = "PNEUMONIA" if pred > novo_limiar else "NORMAL"
            
            # Exibir imagem com a predição
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 5, i+1)
            plt.imshow(img)
            plt.title(f"{predicao}\nConfiança: {pred:.2f}")
            plt.axis("off")
            i += 1
plt.show()
