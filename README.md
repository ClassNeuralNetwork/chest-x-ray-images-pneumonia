# 🫁 Projeto: Classificação de Pneumonia em Radiografias de Tórax com DenseNet121 e DenseNet169 🩻

## 📚 Objetivo da atividade

Projeto desenvolvido como parte de atividades acadêmicas de Redes Neurais Artificiais, englobando a implementação comparativa de múltiplas abordagens (DenseNet121 e DenseNet169) e a seleção fundamentada da melhor arquitetura para o desenvolvimento do artigo científico.

- **Base de dados:** Chest X-Ray Images (Pneumonia)
- **Problema:** Classificação de Imagens Médicas (Binária)
- **Modelos Implementados:** DenseNet121 e DenseNet169
- **Modelo Escolhido:** DenseNet121 (Transfer Learning / Fine-Tuning)
- **Framework:** TensorFlow/Keras

---

## 📝 Visão Geral

Este projeto foi desenvolvido utilizando Aprendizado Profundo (*Deep Learning*), aplicando uma Rede Neural Convolucional (CNN) baseada na arquitetura **DenseNet121** (com Transfer Learning) para classificar imagens de radiografia de tórax de pacientes.

O modelo foi treinado para realizar uma **classificação binária**, identificando duas possibilidades:

- `NORMAL` → Sem sinais de pneumonia
- `PNEUMONIA` → Com presença de padrões sugestivos de pneumonia

O projeto envolve o carregamento dinâmico dos dados via KaggleHub, pré-processamento e normalização das imagens, cálculo de pesos de classes para lidar com o desbalanceamento, aplicação de técnicas de *Data Augmentation*, construção e ajuste fino (*fine-tuning*) da arquitetura DenseNet121, além da avaliação detalhada do modelo utilizando métricas de classificação, matriz de confusão e explicabilidade com **SHAP**.

---

## 🎯 Problema

O problema abordado neste projeto é uma tarefa de **classificação binária de imagens médicas**.

O objetivo é utilizar imagens de raio-x de tórax para que uma rede neural profunda consiga classificar cada exame corretamente em uma das duas categorias:

**NORMAL** ou **PNEUMONIA**.

A detecção automatizada de pneumonia a partir de imagens radiográficas auxilia na triagem rápida de pacientes, servindo como ferramenta de suporte à decisão clínica.

---

## 📊 Dataset

O conjunto de dados utilizado neste projeto é o **Chest X-Ray Images (Pneumonia)**, obtido por meio da plataforma Kaggle.

Dataset utilizado:

- **Nome:** Chest X-Ray Images (Pneumonia)
- **Fonte:** Kaggle
- **Identificador no KaggleHub:** `paultimothymooney/chest-x-ray-pneumonia`
- **Estrutura:** Diretórios divididos em treino (`train`) e teste/validação (`test`), contendo subpastas para cada classe (`NORMAL` e `PNEUMONIA`).

O conjunto de dados contém milhares de imagens de raios-X em alta resolução de pacientes pediátricos. Devido ao desbalanceamento natural entre as classes no dataset, foram calculados pesos de classes (*class weights*) para penalizar erros na classe majoritária/minoritária adequadamente durante o treinamento.

---

## ⚙️ Pré-processamento e Aumentação de Dados

Antes e durante o treinamento da rede neural, os dados passaram pelas seguintes etapas:

1. Redimensionamento das imagens para o formato padrão de `224 x 224` pixels;
2. Configuração de lote (*Batch Size*) de `32` imagens;
3. Normalização dos pixels dividindo por `255.0` (convertendo o intervalo para `[0, 1]`);
4. Cálculo de pesos de classes (`compute_class_weight`) para balancear o aprendizado;
5. Aplicação de *Data Augmentation* no pipeline de treino (rotações e zooms aleatórios leves para aumentar a robustez do modelo).

Os conjuntos de dados utilizados foram:
- **Treino:** ~5.216 imagens
- **Validação/Teste:** 624 imagens

---

## 🧠 Modelo de Rede Neural

Foi utilizada a arquitetura pré-treinada **DenseNet121** (com pesos do ImageNet) combinada com camadas densas personalizadas (*Transfer Learning* e *Fine-Tuning parcial*).

A arquitetura final do modelo estruturado foi:

```text
Entrada (224 x 224 x 3)
   ↓
Data Augmentation (RandomRotation, RandomZoom)
   ↓
DenseNet121 (Base pré-treinada, últimas 30 camadas treináveis)
   ↓
GlobalAveragePooling2D
   ↓
Dense (256 neurônios, ReLU + L2 Regularization + Dropout 0.5)
   ↓
Dense (128 neurônios, ReLU + L2 Regularization + Dropout 0.5)
   ↓
Dense (1 neurônio, Sigmoid)
   ↓
Classificação binária (NORMAL / PNEUMONIA)
``` 

### Configurações do treinamento

- **Otimizador:** Adam (`learning_rate = 0.00001`)
- **Função de perda:** Binary Crossentropy
- **Métrica:** Accuracy
- **Épocas:** 50 (com parada antecipada - *Early Stopping* com paciência de 5 épocas)
- **Callback:** `EarlyStopping` (restaurando os melhores pesos)

---

## 📈 Avaliação do modelo

Após o treinamento, o modelo foi avaliado utilizando o conjunto de teste/validação.

Foram obtidas as seguintes métricas gerais:
- **Acurácia global:** `91.19%`

### Relatório de Classificação Detalhado

| Classe | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **NORMAL** | 0.9453 | 0.8120 | 0.8736 | 234 |
| **PNEUMONIA** | 0.8960 | 0.9718 | 0.9323 | 390 |

Além do relatório numérico, foi gerada e salva a **Matriz de Confusão** para visualizar os acertos e erros de falsos-positivos e falsos-negativos.

---

## 📉 Histórico de Aprendizado

Durante o treinamento, foram monitoradas as curvas de perda (*loss*) e acurácia (*accuracy*) para os conjuntos de treino e validação. Os gráficos gerados permitem acompanhar a convergência do modelo e evitar episódios severos de *overfitting* graças à regularização L2, Dropout e estratégias de *Early Stopping*.

O gráfico de desempenho foi salvo localmente como:
```text
Desempenho_DenseNet121.png
```

## 🔎 Explicabilidade com SHAP

Para interpretar as decisões tomadas pela rede neural em imagens médicas, foi aplicada a biblioteca **SHAP (SHapley Additive exPlanations)** utilizando o `GradientExplainer`.

O SHAP permitiu gerar mapas de atribuição visual nas radiografias de teste, destacando quais regiões da imagem de raio-X pesaram mais para a decisão do modelo em classificar o exame como `NORMAL` ou `PNEUMONIA`.

Os gráficos gerados foram salvos como:
```text
shap_amostra_0.png, shap_amostra_1.png, shap_amostra_2.png e shap_amostra_3.png
```
## 💾 Modelo Salvo

O modelo treinado foi serializado e salvo no formato `.keras`:
- `modelDenseNet121_Pneumonia.keras`

---

## 🛠️ Tecnologias utilizadas

- **Python**
- **TensorFlow / Keras**
- **Pandas & NumPy**
- **Scikit-learn**
- **Matplotlib & Seaborn**
- **SHAP**
- **KaggleHub**

---

## 👥 Integrantes do Projeto

Projeto desenvolvido por estudantes da Universidade Federal Rural do Semi-Árido (UFERSA), Campus Pau dos Ferros, como parte da disciplina de Redes Neurais Artificiais.

- **Amanda de Queiroz Santiago**
- **Kayc Henderson Morais Leite**

---
