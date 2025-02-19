# README.md

## **Reconhecimentos e Direitos Autorais**

@autor: JUDSON RODRIGUES CIRIBELLI, GABRIEL FELIPE CARVALHO SILVA, GIORDANO BRUNO DE ARAUJO MOCHEL  
@contato: [Seus Emails - se quiserem]  
@data última versão: [Data de Hoje]  
@versão: 1.0  
@outros repositórios: [URLs - apontem para os seus Gits, se quiserem]  
@Agradecimentos: Universidade Federal do Maranhão (UFMA), Professor Doutor Thales Levi Azevedo Valente, e colegas de curso.

---

Detecção de Pneumonia em Raios-X com Transfer Learning (VGG16)

Descrição do Projeto

Este projeto tem como objetivo a detecção de pneumonia em imagens de raio-X utilizando redes neurais convolucionais (CNNs). A abordagem utilizada foi Transfer Learning com a arquitetura VGG16, que foi pré-treinada no ImageNet e ajustada para a classificação binária (NORMAL vs PNEUMONIA).

A detecção precoce de pneumonia é um desafio clínico relevante, e o uso de inteligência artificial aplicada à radiologia pode ajudar médicos e especialistas a realizarem diagnósticos mais rápidos e eficientes. Este projeto implementa um modelo treinado para essa finalidade, avaliando seu desempenho em um dataset público de imagens pulmonares.

Metodologia

O projeto segue um fluxo estruturado para a construção e avaliação do modelo de detecção:

1️⃣ Pré-processamento do Dataset

- Utilização do dataset Chest X-Ray Dataset, que contém imagens de raio-X divididas em duas classes:

- NORMAL: Pacientes sem pneumonia.

- PNEUMONIA: Pacientes diagnosticados com pneumonia.

- Conversão das imagens para escala de cinza (se necessário) e redimensionamento para 224x224 pixels.

- Normalização dos valores dos pixels para ficarem na escala de 0 a 1.

2️⃣ Construção do Modelo

- Utilização da arquitetura VGG16 como base do modelo (Transfer Learning).

- Descongelamento das últimas camadas convolucionais para Fine-Tuning.

- Adição de camadas densas personalizadas para a classificação binária.

- Treinamento com Binary Crossentropy para melhorar o equilíbrio da classificação.

3️⃣ Treinamento e Avaliação

- Utilização de Data Augmentation para aumentar a diversidade dos dados e melhorar a generalização.

- Aplicação de pesos balanceados para classes, evitando viés excessivo para a classe mais comum.

- Validação do modelo utilizando uma fração do dataset.

- Avaliação da performance com acurácia, precisão, recall e matriz de confusão.

4️⃣ Validação do Modelo em Novas Imagens

- O modelo é testado em imagens externas não presentes no treinamento.

- As predições são analisadas para verificar a capacidade de generalização do modelo.

- Aplicação do Grad-CAM para visualizar as regiões das imagens que mais influenciaram as decisões do modelo.

## **Copyright/License**

Este material é resultado de um trabalho acadêmico para a disciplina **INTELIGÊNCIA ARTIFICIAL**, sob a orientação do professor **Dr. THALES LEVI AZEVEDO VALENTE**, semestre letivo **2024.2**, curso **Engenharia da Computação**, na **Universidade Federal do Maranhão (UFMA)**.

Todo o material sob esta licença é **software livre**: pode ser usado para fins **acadêmicos e comerciais** sem nenhum custo. Não há papelada, nem royalties, nem restrições de "copyleft" do tipo GNU. Ele é licenciado sob os termos da **Licença MIT**, conforme descrito abaixo, e, portanto, é compatível com a **GPL** e também se qualifica como **software de código aberto**. É de **domínio público**.

O espírito desta licença é que você é **livre para usar este material para qualquer finalidade, sem nenhum custo**. O único requisito é que, se você usá-los, **nos dê crédito**.

---

## **Licença MIT**

Licenciado sob a **Licença MIT**. Permissão é concedida, gratuitamente, a qualquer pessoa que obtenha uma cópia deste software e dos arquivos de documentação associados (o "Software"), para lidar no Software sem restrição, incluindo sem limitação os direitos de **usar, copiar, modificar, mesclar, publicar, distribuir, sublicenciar e/ou vender cópias do Software**, e permitir pessoas a quem o Software é fornecido a fazê-lo, sujeito às seguintes condições:

Este aviso de direitos autorais e este aviso de permissão devem ser incluídos em todas as cópias ou partes substanciais do Software.

O SOFTWARE É FORNECIDO "COMO ESTÁ", **SEM GARANTIA DE QUALQUER TIPO, EXPRESSA OU IMPLÍCITA**, INCLUINDO MAS NÃO SE LIMITANDO ÀS GARANTIAS DE **COMERCIALIZAÇÃO, ADEQUAÇÃO A UM DETERMINADO FIM E NÃO INFRINGÊNCIA**. **EM NENHUM CASO OS AUTORES OU DETENTORES DE DIREITOS AUTORAIS SERÃO RESPONSÁVEIS POR QUALQUER RECLAMAÇÃO, DANOS OU OUTRA RESPONSABILIDADE, SEJA EM AÇÃO DE CONTRATO, TORT OU OUTRA FORMA, DECORRENTE DE, FORA DE OU EM CONEXÃO COM O SOFTWARE OU O USO OU OUTRAS NEGOCIAÇÕES NO SOFTWARE.**

Para mais informações sobre a Licença MIT: **[https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)**
