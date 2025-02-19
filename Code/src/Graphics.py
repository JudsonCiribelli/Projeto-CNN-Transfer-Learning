import matplotlib.pyplot as plt

# Definição das métricas
metrics = ["Precision", "Recall", "F1-Score"]

# Valores das métricas para cada modelo
vgg16_normal = [0.89, 0.72, 0.80]  # NORMAL
vgg16_pneumonia = [0.85, 0.95, 0.90]  # PNEUMONIA

mobilenet_normal = [0.98, 0.74, 0.84]  # NORMAL
mobilenet_pneumonia = [0.87, 0.99, 0.92]  # PNEUMONIA

# Criando os gráficos
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico de Linhas para o VGG16
ax[0].plot(metrics, vgg16_normal, marker='o', linestyle='-', label="NORMAL", color='blue')
ax[0].plot(metrics, vgg16_pneumonia, marker='s', linestyle='--', label="PNEUMONIA", color='red')
ax[0].set_title("Desempenho do VGG16")
ax[0].set_ylim(0, 1)
ax[0].set_ylabel("Valor")
ax[0].set_xlabel("Métricas")
ax[0].legend()

# Gráfico de Linhas para o MobileNetV2
ax[1].plot(metrics, mobilenet_normal, marker='o', linestyle='-', label="NORMAL", color='blue')
ax[1].plot(metrics, mobilenet_pneumonia, marker='s', linestyle='--', label="PNEUMONIA", color='red')
ax[1].set_title("Desempenho do MobileNetV2")
ax[1].set_ylim(0, 1)
ax[1].set_ylabel("Valor")
ax[1].set_xlabel("Métricas")
ax[1].legend()

# Ajustando o layout
plt.tight_layout()

# Exibir os gráficos
plt.show()