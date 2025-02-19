import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Definições
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

# Caminhos do dataset
dataset_path = "dataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")
val_path = os.path.join(dataset_path, "val")

# Pré-processamento e Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

treinamento = datagen.flow_from_directory(train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
validacao = datagen.flow_from_directory(val_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
teste = datagen.flow_from_directory(test_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

# Calcular os Pesos das Classes
total_classes = list(treinamento.class_indices.keys())
num_samples = treinamento.classes
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(num_samples), y=num_samples)
class_weight_dict = {i: class_weights[i] for i in range(len(total_classes))}
print(f"Pesos de classe ajustados: {class_weight_dict}")

# Carregar MobileNetV2 pré-treinado
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True  # Descongelando as últimas camadas para Fine-Tuning

# Liberar as últimas 5 camadas convolucionais para treinamento
for layer in base_model.layers[:-5]:
    layer.trainable = False

# Construção do modelo
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
out = Dense(1, activation='sigmoid')(x)  # Saída binária (NORMAL vs PNEUMONIA)

model = Model(inputs=base_model.input, outputs=out)

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
              loss=BinaryCrossentropy(), 
              metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Early Stopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinamento do modelo
history = model.fit(
    treinamento,
    validation_data=validacao,
    epochs=EPOCHS,
    class_weight=class_weight_dict,  # Aplicando os pesos balanceados
    callbacks=[early_stopping],
    verbose=1
)

# Salvar o modelo treinado
model.save("models/mobilenetv2_pneumonia_finetuned.h5")

# Avaliação no conjunto de teste
eval_result = model.evaluate(teste)
print(f"Loss no teste: {eval_result[0]:.4f} | Acurácia no teste: {eval_result[1] * 100:.2f}%")