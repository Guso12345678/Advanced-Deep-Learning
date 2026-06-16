# Machine Perception and Learning — Prácticas de Deep Learning
 
Colección de laboratorios de la asignatura **Machine Perception and Learning** (IMAT 4º — Comillas ICAI, curso 2025/26).  
Implementación desde cero en PyTorch de los principales modelos de Deep Learning moderno: redes convolucionales, recurrentes, transformers, autoencoders (AE, DAE, VAE), GANs (DCGAN, WGAN-GP), Few-Shot Learning con Prototypical Networks, Reinforcement Learning con PPO, modelos de difusión (DDPM) y Graph Neural Networks.
 
---
 
## Estructura del repositorio
 
```
Machine Perception and Learning/
│
├── Práctica 1/          # Reglas y Regresión Logística
├── Práctica 2/          # MLP en PyTorch y NumPy (backpropagation manual)
├── Practica 3/          # LeNet y VGG
├── Practica_4_Res_Net/  # ResNet (conexiones residuales)
├── Practica_5_RNNs/     # RNN (redes recurrentes)
├── Practica_6_Transformers/  # Transformer (atención y encoders)
├── Practica_7_Regularizacion/ # Técnicas de regularización + recomendador
│
├── 07_fewshot-*/        # Few-Shot Learning — Prototypical Networks (Omniglot)
├── 08_ae-*/             # Autoencoders: AE, DAE y VAE (MNIST)
├── 09_gan-*/            # GANs: DCGAN y WGAN-GP (MNIST)
├── 10_rl-*/             # Reinforcement Learning: PPO (CartPole)
├── 11_diffusion-*/      # Modelos de difusión: DDPM
└── 12_gnn-*/            # Graph Neural Networks (GNN)
```
 
---
 
## Contenido por práctica
 
### Práctica 1 — Clasificación con reglas y Regresión Logística
Implementación de clasificadores basados en reglas y regresión logística como introducción al pipeline de ML.
 
### Práctica 2 — MLP en PyTorch y NumPy
Perceptrón multicapa implementado de dos formas: con la API de PyTorch y con NumPy desde cero, incluyendo backpropagation manual para entender el grafo computacional.
 
### Práctica 3 — LeNet y VGG
Implementación de las arquitecturas clásicas de redes convolucionales: LeNet-5 y VGGNet, con entrenamiento en datasets estándar.
 
### Práctica 4 — ResNet
Implementación de ResNet con **conexiones residuales** (skip connections) para resolver el problema del desvanecimiento del gradiente en redes profundas.
 
### Práctica 5 — RNN
Redes recurrentes para modelado de secuencias. Implementación de RNN estándar con aplicaciones de predicción secuencial.
 
### Práctica 6 — Transformers
Implementación del mecanismo de **self-attention** y del encoder Transformer completo, incluyendo positional encoding y multi-head attention.
 
### Práctica 7 — Regularización y sistema de recomendación
Técnicas de regularización (Dropout, BatchNorm, L1/L2) y un sistema de recomendación basado en embeddings.
 
---
 
### Lab 07 — Few-Shot Learning: Prototypical Networks
 
**Dataset:** Omniglot (19.280 imágenes de 964 caracteres de 30 alfabetos)  
**Tarea:** N-way K-shot classification — aprender a clasificar desde muy pocos ejemplos
 
**Arquitectura:** CNN de 4 bloques (`Conv2d → BatchNorm → ReLU → MaxPool`) + capa FC de 64 dimensiones.
 
El enfoque meta-learning entrena episodios de N-way K-shot: para cada episodio se calculan los **prototipos** de cada clase (media de los embeddings del support set) y se clasifica cada query point por distancia euclídea al prototipo más cercano:
 
```python
# Distancia negativa como logit para cross-entropy
distances = -torch.cdist(query_embeddings, prototypes)
loss = F.cross_entropy(distances, query_labels)
```
 
**Modelos guardados:** `1_pn.pth`
 
---
 
### Lab 08 — Autoencoders: AE, DAE y VAE
 
**Dataset:** MNIST (28×28 dígitos escritos a mano)
 
**AE — Autoencoder estándar:**
- Encoder: `784 → 1000 → 500 → 250 → n_components` (ReLU)
- Decoder: `n_components → 250 → 500 → 1000 → 784`
- Pérdida: MSE de reconstrucción
- Bottleneck de 10 dimensiones
**DAE — Denoising Autoencoder:**
Misma arquitectura pero entrenado con imágenes con ruido gaussiano como entrada y la imagen limpia como objetivo — aprende representaciones robustas.
 
**VAE — Variational Autoencoder:**
El encoder genera media `μ` y log-varianza `log σ²`. El espacio latente se muestrea con el **reparameterization trick**:
 
```python
z = μ + σ * ε,  ε ~ N(0, I)
```
 
Función de pérdida: reconstrucción + KL divergence:
```
L = MSE(x, x̂) + β · KL(N(μ, σ²) || N(0, I))
```
 
**Modelos guardados:** `1_ae_encoder.pth`, `1_ae_decoder.pth`, `1_ae_embeddings.pth`, `2_dae.pth`, `3_vae_encoder.pth`, `3_vae_decoder.pth`
 
---
 
### Lab 09 — GANs: DCGAN y WGAN-GP
 
**Dataset:** MNIST
 
**DCGAN — Deep Convolutional GAN:**
 
- **Generator:** `z (latente, dim=10) → ConvTranspose2d × 4 → imagen 28×28`
- **Discriminator:** `Conv2d × 4 → LeakyReLU(0.2) → probabilidad real/falso`
- Pérdida: Binary Cross-Entropy
```python
# Loss del generador
loss_G = BCE(D(G(z)), real_labels)
# Loss del discriminador
loss_D = BCE(D(x_real), 1) + BCE(D(G(z)), 0)
```
 
**WGAN-GP — Wasserstein GAN con Gradient Penalty:**  
Reemplaza la BCE por la **distancia de Wasserstein**, que proporciona gradientes más estables. El critic (discriminator sin sigmoid) se entrena más veces que el generador. Se añade una **gradient penalty** para imponer la condición de Lipschitz:
 
```
L = E[D(fake)] - E[D(real)] + λ · E[(||∇D(x̂)||₂ - 1)²]
```
 
Evaluación con **Fréchet Inception Distance (FID)** usando el archivo `fd.py`.
 
**Modelos guardados:** `1_dcgan_g.pth`, `1_dcgan_d.pth`, `2_wgan_g.pth`, `2_wgan_d.pth`
 
---
 
### Lab 10 — Reinforcement Learning: PPO
 
**Entorno:** CartPole-v1 (Gymnasium)  
**Algoritmo:** Proximal Policy Optimization (PPO) con GAE
 
**Arquitecturas:**
- **PolicyNetwork:** `state_dim → 128 → 128 → action_dim` (Softmax → distribución Categorical)
- **ValueNetwork:** `state_dim → 128 → 128 → 1` (estimador de retorno)
**Generalized Advantage Estimation (GAE)** — reduce varianza en el gradiente de política:
```python
delta_t = r_t + γ · V(s_{t+1}) · mask - V(s_t)
A_t = delta_t + γ · λ · mask · A_{t+1}
```
 
**PPO clipping** — limita el cambio de política para estabilidad:
```python
ratio = exp(new_log_prob - old_log_prob)
loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```
 
Colecta 2048 transiciones por iteración, actualiza con mini-batches de 32.
 
**Modelos guardados:** `1_ppo_cartpole.pth`
 
---
 
### Lab 11 — Modelos de Difusión: DDPM
 
**Denoising Diffusion Probabilistic Model** — genera imágenes revirtiendo un proceso de difusión de ruido gaussiano.
 
- **Forward process:** añade ruido gradualmente en T pasos: `q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)`
- **Reverse process:** una red U-Net aprende a predecir el ruido para deshacer el proceso
---
 
### Lab 12 — Graph Neural Networks (GNN)
 
Implementación de redes neuronales sobre grafos para clasificación de nodos/grafos.
 
---
 
## Tecnologías
 
- Python 3.9 / PyTorch
- `torchvision` — datasets (MNIST, Omniglot) y transformaciones
- `gymnasium` — entorno CartPole para RL
- `matplotlib` / `seaborn` — visualización
- Jupyter Notebook
---
 
## Autor
 
**Guzman Ignacio Perez Ibarz**
 
Laboratorios de la asignatura Machine Perception and Learning, 4º Ingeniería Matemática e Inteligencia Artificial (IMAT), Comillas ICAI. Curso 2025/26.
