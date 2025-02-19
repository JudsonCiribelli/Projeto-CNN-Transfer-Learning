# Avaliação da Precisão do Modelo VGG16
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Definir caminhos do dataset
dataset_path = "dataset"
test_path = os.path.join(dataset_path, "test")

# Parâmetros para pré-processamento
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Carregar o modelo treinado
model_path = "models/vgg16_pneumonia_finetuned_v2.h5"
model = tf.keras.models.load_model(model_path)

# Pré-processamento do conjunto de teste
datagen = ImageDataGenerator(rescale=1./255)
teste = datagen.flow_from_directory(test_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

# Fazer previsões
predictions = model.predict(teste)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = teste.classes
labels = list(teste.class_indices.keys())

# Gerar relatório de classificação
print("Relatório de Classificação:")
print(classification_report(true_classes, predicted_classes, target_names=labels))
