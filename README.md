# 🚀 MPPoGPUS

**Massively Parallel Programming on GPUs**

[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17-00599C?logo=cplusplus)](https://isocpp.org/)

---

## 📋 Descrizione

Questo repository contiene **script, implementazioni ed esercitazioni** per il corso di **Massively Parallel Programming on GPUs**. Il progetto raccoglie esempi didattici e sperimentazioni in **C/C++/CUDA** per illustrare tecniche di programmazione massivamente parallela su GPU NVIDIA.

---

## 📂 Struttura del Repository

```
MPPoGPUS/
├── vecAdd/           # Esempio base: somma vettoriale su GPU
├── vecAddErrors/     # Varianti con gestione e diagnostica degli errori
├── blur/             # Implementazioni per applicare blur a immagini
├── grayscale/        # Conversione immagini in scala di grigi
├── librerie/         # Librerie helper (stb_image, stb_image_write)
├── .vscode/          # Configurazioni editor VS Code
└── README.md         # Questo file
```

---

## 🛠️ Requisiti

Prima di iniziare, assicurati di avere i seguenti strumenti installati:

| Requisito | Descrizione |
|-----------|-------------|
| **GPU NVIDIA** | GPU compatibile con CUDA |
| **CUDA Toolkit** | Versione 11.x o successiva |
| **nvcc** | Compilatore NVIDIA CUDA |
| **g++** | Toolchain C/C++ |
| **Sistema Operativo** | Linux / WSL / macOS con driver NVIDIA |

### Verifica Installazione

```bash
# Verifica versione CUDA
nvcc --version

# Verifica GPU disponibile
nvidia-smi

# Verifica compilatore C++
g++ --version
```

---

## 🚀 Quick Start

### 1. Clona il Repository

```bash
git clone https://github.com/Chrustho/MPPoGPUS.git
cd MPPoGPUS
```

### 2. Compila ed Esegui un Esempio

```bash
# Esempio: Vector Addition
cd vecAdd
nvcc -o vecAdd vecAdd.cu
./vecAdd
```

---

## 📚 Contenuti

### 🔢 Vector Addition (`vecAdd/`)
Esempio fondamentale di programmazione CUDA che dimostra la somma di due vettori eseguita in parallelo sulla GPU.

**Concetti chiave:**
- Allocazione memoria device (`cudaMalloc`)
- Trasferimento dati host ↔ device (`cudaMemcpy`)
- Lancio kernel con configurazione griglia/blocchi
- Gestione della memoria GPU

### ⚠️ Error Handling (`vecAddErrors/`)
Varianti dell'esempio vector addition con focus sulla gestione degli errori CUDA.

**Concetti chiave:**
- Macro per error checking
- `cudaGetLastError()` e `cudaDeviceSynchronize()`
- Debug e diagnostica

### 🖼️ Image Processing

#### Blur (`blur/`)
Implementazione di filtri di sfocatura su immagini utilizzando convoluzioni parallele.

**Concetti chiave:**
- Convoluzioni 2D su GPU
- Gestione immagini con stb_image
- Memoria shared per ottimizzazione

#### Grayscale (`grayscale/`)
Conversione di immagini a colori in scala di grigi.

**Concetti chiave:**
- Manipolazione pixel parallela
- Formule di luminanza
- I/O immagini

---

## 📖 Librerie Utilizzate

Il progetto utilizza le seguenti librerie header-only per la gestione delle immagini:

| Libreria | Descrizione |
|----------|-------------|
| **stb_image.h** | Caricamento immagini (PNG, JPG, BMP, etc.) |
| **stb_image_write.h** | Salvataggio immagini in vari formati |

Queste librerie sono incluse nella cartella `librerie/` e non richiedono installazione aggiuntiva.

---

## 💡 Suggerimenti per lo Sviluppo

### Compilazione Ottimizzata

```bash
# Compilazione con ottimizzazioni
nvcc -O3 -arch=sm_XX -o output input.cu

```

### Debug

```bash
# Compilazione con simboli di debug
nvcc -g -G -o output_debug input.cu

# Utilizzo di cuda-memcheck
cuda-memcheck ./output_debug
```

---

## 📝 Risorse Utili

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [stb Libraries](https://github.com/nothings/stb)

---

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file [LICENSE](LICENSE) per i dettagli.

---

## 👤 Autore

**Chrustho** - [GitHub](https://github.com/Chrustho)

---
