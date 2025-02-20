import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import os
import numpy as np

# Definições
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50  # Aumentamos para melhor aprendizado
LEARNING_RATE = 0.0001  # Ajuste fino da taxa de aprendizado
ALPHA = 0.25  # Parâmetro do Focal Loss
GAMMA = 2.0  # Parâmetro do Focal Loss

# Caminhos do dataset
dataset_path = "dataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")
val_path = os.path.join(dataset_path, "val")

# Ajustando pesos das classes manualmente para reduzir viés
peso_normal = 1.2  # Aumentamos o peso da classe NORMAL
peso_pneumonia = 1.0
class_weights = {0: peso_normal, 1: peso_pneumonia}
print("Pesos ajustados para as classes:", class_weights)

# Pré-processamento e Data Augmentation aprimorado
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],  # Ajuste de brilho para melhorar robustez
    horizontal_flip=True,
    fill_mode='nearest'
)

treinamento = datagen.flow_from_directory(train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
validacao = datagen.flow_from_directory(val_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
teste = datagen.flow_from_directory(test_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

# Carregar VGG16 pré-treinado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True  # Descongelando mais camadas para Fine-Tuning

# Liberar as últimas 4 camadas convolucionais para treinamento
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Construir modelo personalizado
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)  # Adicionando Batch Normalization para estabilidade
x = Dropout(0.6)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
out = Dense(1, activation='sigmoid')(x)  # Saída binária (NORMAL vs PNEUMONIA)

model = Model(inputs=base_model.input, outputs=out)

# Compilar o modelo com novo otimizador e Focal Loss
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
              loss=BinaryFocalCrossentropy(alpha=ALPHA, gamma=GAMMA), 
              metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Early Stopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinamento do modelo com ajuste de pesos das classes
history = model.fit(
    treinamento,
    validation_data=validacao,
    epochs=EPOCHS,
    class_weight=class_weights,  # Aplicando pesos ajustados
    callbacks=[early_stopping],  # Aplicando Early Stopping
    verbose=1
)

# Salvar o modelo treinado
model.save("Code/models/vgg16_pneumonia_finetuned_v2.h5")

# Avaliação no conjunto de teste
eval_result = model.evaluate(teste)
print(f"Loss no teste: {eval_result[0]:.4f} | Acurácia no teste: {eval_result[1] * 100:.2f}%")
