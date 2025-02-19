import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Definir caminhos
dataset_path = "dataset"
test_path = os.path.join(dataset_path, "test")
model_path = "models/mobilenetv2_pneumonia_finetuned.h5"

# Carregar modelo treinado
model = tf.keras.models.load_model(model_path)

# Selecionar uma imagem de teste aleatória
def escolher_imagem_aleatoria(diretorio):
    classes = os.listdir(diretorio)
    classe_escolhida = np.random.choice(classes)
    caminho_classe = os.path.join(diretorio, classe_escolhida)
    imagem_escolhida = np.random.choice(os.listdir(caminho_classe))
    return os.path.join(caminho_classe, imagem_escolhida), classe_escolhida

img_path, real_class = escolher_imagem_aleatoria(test_path)
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Fazer previsão
preds = model.predict(img_array)
predicted_class = "PNEUMONIA" if preds[0][0] > 0.5 else "NORMAL"

# Gerar Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Nome da última camada convolucional do MobileNet
last_conv_layer_name = "block5_conv3"
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# Superimpor Grad-CAM à imagem original
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# Exibir imagem original e Grad-CAM
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Imagem Original ({real_class})")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Grad-CAM ({predicted_class})")
plt.axis("off")

plt.show()
