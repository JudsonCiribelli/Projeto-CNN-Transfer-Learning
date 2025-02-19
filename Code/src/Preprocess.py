import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Caminho para as pastas do dataset
dataset_path = "dataset/"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")
val_path = os.path.join(dataset_path, "val")

# Parâmetros para pré-processamento
IMG_SIZE = (224, 224)  # Tamanho das imagens para o modelo
BATCH_SIZE = 32  # Número de imagens processadas por vez

# Função para carregar e exibir imagens
def visualizar_amostras(diretorio, num_imagens=5):
    classes = os.listdir(diretorio)
    fig, axes = plt.subplots(1, num_imagens, figsize=(15, 5))
    for i, classe in enumerate(classes):
        caminho_classe = os.path.join(diretorio, classe)
        imagens = os.listdir(caminho_classe)
        img_path = os.path.join(caminho_classe, imagens[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter para RGB
        img = cv2.resize(img, IMG_SIZE)  # Redimensionar
        axes[i].imshow(img)
        axes[i].set_title(classe)
        axes[i].axis("off")
    plt.show()

# Visualizar algumas imagens de treino
print("Amostras do conjunto de treino:")
visualizar_amostras(train_path)

# Normalização e Data Augmentation
datagen = ImageDataGenerator(rescale=1./255)

treinamento = datagen.flow_from_directory(train_path,
                                          target_size=IMG_SIZE,
                                          batch_size=BATCH_SIZE,
                                          class_mode='binary')

validacao = datagen.flow_from_directory(val_path,
                                        target_size=IMG_SIZE,
                                        batch_size=BATCH_SIZE,
                                        class_mode='binary')

teste = datagen.flow_from_directory(test_path,
                                     target_size=IMG_SIZE,
                                     batch_size=BATCH_SIZE,
                                     class_mode='binary')

# Verificar balanceamento das classes
print("Distribuição das classes no conjunto de treino:")
labels, counts = np.unique(treinamento.classes, return_counts=True)
plt.bar(labels, counts, tick_label=["NORMAL", "PNEUMONIA"], color=['blue', 'red'])
plt.xlabel("Classe")
plt.ylabel("Número de Imagens")
plt.title("Distribuição das Classes no Conjunto de Treinamento")
plt.show()
