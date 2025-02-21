import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np

# Definições
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30  # Aumentamos para tentar melhor aprendizado
LEARNING_RATE = 0.001  # Taxa de aprendizado reduzida para melhor estabilidade
ALPHA = 0.25
GAMMA = 2.0

# Caminhos do dataset
dataset_path = "dataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")
val_path = os.path.join(dataset_path, "val")

# Data Augmentation aprimorado
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Aumentamos a variação de rotação
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.4,  # Maior variação no zoom
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],  # Aumento da variação de brilho
    fill_mode='nearest'
)

# Carregar os dados
treinamento = datagen.flow_from_directory(train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
validacao = datagen.flow_from_directory(val_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
teste = datagen.flow_from_directory(test_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

# **Ajuste automático dos pesos das classes**
y_train = treinamento.classes  # Obtendo as classes do conjunto de treino
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Pesos ajustados automaticamente:", class_weights_dict)

# Carregar VGG16 pré-treinado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True  # Descongelando mais camadas para Fine-Tuning

# Liberar mais camadas para aprendizado
for layer in base_model.layers[:-6]:  # Antes estava -10, agora liberamos mais camadas
    layer.trainable = False

# Construção do modelo
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)  # Batch Normalization para maior estabilidade
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)  # Saída binária

model = Model(inputs=base_model.input, outputs=out)

# Compilar o modelo com novo otimizador e Focal Loss
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
              loss=BinaryFocalCrossentropy(alpha=ALPHA, gamma=GAMMA), 
              metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Early Stopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# **Treinamento do modelo**
history = model.fit(
    treinamento,
    validation_data=validacao,
    epochs=EPOCHS,
    class_weight=class_weights_dict,  # Aplicando pesos ajustados automaticamente
    callbacks=[early_stopping],
    verbose=1
)

# **Salvar o modelo treinado**
model.save("Code/models/vgg16_pneumonia_finetuned_v3.h5")

# **Avaliação no conjunto de teste**
eval_result = model.evaluate(teste)
print(f"Loss no teste: {eval_result[0]:.4f} | Acurácia no teste: {eval_result[1] * 100:.2f}%")
