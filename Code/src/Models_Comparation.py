import numpy as np
import matplotlib.pyplot as plt

# Valores das métricas obtidas pelo classification_report (substitua pelos seus resultados)
metrics = ["Precision", "Recall", "F1-Score"]

# MÉTRICAS VGG16 (Substitua pelos seus resultados)
vgg16_normal = [0.98, 0.74, 0.84]  # [Precision, Recall, F1-Score] para NORMAL
vgg16_pneumonia = [0.87, 0.99, 0.92]  # [Precision, Recall, F1-Score] para PNEUMONIA

# MÉTRICAS MobileNetV2 (Substitua pelos seus resultados)
mobilenet_normal = [0.85, 0.85, 0.84]  # [Precision, Recall, F1-Score] para NORMAL
mobilenet_pneumonia = [0.91, 0.91, 0.91]  # [Precision, Recall, F1-Score] para PNEUMONIA

# Criando a figura e os subgráficos
fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # 1 linha, 2 colunas

# Posições das barras
x = np.arange(len(metrics))

# Largura das barras
width = 0.3

# 📌 Gráfico para a classe NORMAL
ax[0].bar(x - width/2, vgg16_normal, width, label="VGG16", color='blue', alpha=0.7)
ax[0].bar(x + width/2, mobilenet_normal, width, label="MobileNetV2", color='orange', alpha=0.7)
ax[0].set_xticks(x)
ax[0].set_xticklabels(metrics)
ax[0].set_ylim(0, 1)
ax[0].set_title("Desempenho para NORMAL")
ax[0].set_ylabel("Valor")
ax[0].legend()

# 📌 Gráfico para a classe PNEUMONIA
ax[1].bar(x - width/2, vgg16_pneumonia, width, label="VGG16", color='blue', alpha=0.7)
ax[1].bar(x + width/2, mobilenet_pneumonia, width, label="MobileNetV2", color='orange', alpha=0.7)
ax[1].set_xticks(x)
ax[1].set_xticklabels(metrics)
ax[1].set_ylim(0, 1)
ax[1].set_title("Desempenho para PNEUMONIA")
ax[1].set_ylabel("Valor")
ax[1].legend()

# Exibir gráfico
plt.tight_layout()
plt.show()
