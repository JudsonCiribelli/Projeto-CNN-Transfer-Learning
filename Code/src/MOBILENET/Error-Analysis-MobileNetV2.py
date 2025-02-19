import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Definir caminhos do dataset
dataset_path = "dataset"
test_path = os.path.join(dataset_path, "test")

# Parâmetros para pré-processamento
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Carregar o modelo treinado
model_path = "models/mobilenetv2_pneumonia_finetuned.h5"
model = tf.keras.models.load_model(model_path)

# Pré-processamento do conjunto de teste
datagen = ImageDataGenerator(rescale=1./255)
teste = datagen.flow_from_directory(test_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

# Fazer previsões
predictions = model.predict(teste)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = teste.classes
labels = list(teste.class_indices.keys())

# Criar Matriz de Confusão
cm = confusion_matrix(true_classes, predicted_classes)
print("Matriz de Confusão:")
print(cm)

# Relatório de Classificação
print("\nRelatório de Classificação:")
print(classification_report(true_classes, predicted_classes, target_names=labels))

# Identificar erros
misclassified_idx = np.where(predicted_classes != true_classes)[0]

# Exibir algumas imagens que foram classificadas incorretamente
if len(misclassified_idx) > 0:
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()
    for i in range(9):
        if i < len(misclassified_idx):
            img_path = teste.filepaths[misclassified_idx[i]]
            img = plt.imread(img_path)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Real: {labels[true_classes[misclassified_idx[i]]]}\nPrevisto: {labels[predicted_classes[misclassified_idx[i]]]} ")
            axes[i].axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Nenhum erro encontrado.")
