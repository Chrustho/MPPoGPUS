# MPPoGPUS
Scripts e implementazioni per il corso/esercitazioni **Massively Parallel Programming on GPUs**

**Repository:** `Chrustho/MPPoGPUS`. :contentReference[oaicite:1]{index=1}

---

## Descrizione
Raccolta di esempi e sperimentazioni in C/C++/CUDA per illustrare tecniche di programmazione massivamente parallela su GPU.  
Le cartelle principali includono esempi didattici (es. somma vettoriale), esperimenti su immagini (blur, grayscale) e librerie di supporto.

---

## Contenuto del repository
- `vecAdd/` — esempio base di somma vettoriale (vector addition) su GPU.
- `vecAddErrors/` — varianti ed esperimenti con gestione/diagnostica degli errori.
- `blur/` — implementazioni per applicare blur a immagini (probabilmente in CUDA/C++).
- `grayscale/` — conversione di immagini in scala di grigi (esempi GPU).
- `librerie/` — librerie / helper usati dagli esempi.
- `.vscode/` — configurazioni di editor (opzionale).
- `README.md` — questo file.



---

## Requisiti
- GPU NVIDIA compatibile (CUDA)
- NVIDIA CUDA Toolkit (es. 11.x o successivo — adattare alla versione usata)
- `nvcc` (compilatore CUDA)
- `g++` / toolchain C/C++
- (opzionale) OpenCV per gli esempi che manipolano immagini (se gli esempi la richiedono)
- Linux / WSL o altra piattaforma con driver NVIDIA aggiornati

