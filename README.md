# DAR-NN-based-FEM-SCA

In this repository, we propose a Domain Adversarial ReFeture Nueral Network (DAR-NN) to facilitate "noisy-clean" adaptation for far field EM traces captured at long distances.

By collaboratively utilizing two other deep-learning classifiers as regularization terms of the DAE model, the DAR-NN model aims to rebuild key-dependent features for further attacks, thereby achieving a more efficient FEM-SCA. 

The main idea behind the proposed DAR-NN model is to build the mapping from the "noisy" to "groundtruth" traces while using a CNN classifier pre-trained for the attack and a MLP discriminator as regularization terms. However, it is impractical to capture far-field EM traces truly "clean". The training process of the proposed DAR-NN model is described in Fig. 1.

![image](https://github.com/user-attachments/assets/1d053434-33b5-4d22-b896-c504827446be)


At the attack stage, the trained reconstructor in DAR-NN is used independently as the filter to map the captured ”victim” traces to ”reconstructed” traces. Next, the newly generated ”reconstructed” traces are used to derive the secret key of the victim device by using the adapted classifier in DAR-NN (see Fig. 3).

![image](https://github.com/user-attachments/assets/ce7f54bc-953c-4050-9ca5-b19e8bfdcd26)

# DAR-NN: Domain-Adversarial Reconstruction Neural Network for Side-Channel Analysis

This repository contains two scripts:

- **`DAR-NN_train.py`**: Builds and trains a domain-adversarial neural network for side-channel trace classification.
- **`PGE_test.py`**: Evaluates the model using Partial Guessing Entropy (PGE) across multiple devices.

---

## 📁 Directory Structure

```
.
├── DAR-NN_train.py          # Model definition and training script
├── PGE_test.py              # PGE evaluation script
├── Data_processed/
│   ├── Train/
│   │   ├── noisy_traces.npy
│   │   ├── groundtruth_traces.npy
│   │   ├── key_labels.npy
│   │   └── domain_labels.npy
│   └── Test/
│       ├── D6/
│       │   ├── 1_nor_traces_maxmin.npy
│       │   ├── 1_10th_roundkey.npy
│       │   └── 1_ct.npy
│       └── ...
├── DAR-NN_model/
│   └── DAR-NN_4.h5          # Trained model for evaluation
```

---

## 🚀 1. Training the Model (`DAR-NN_train.py`)

### 🔧 Description

- Constructs a three-part neural architecture:
  - A **denoising autoencoder** to reconstruct noisy traces.
  - A **classifier** to predict AES keys.
  - A **domain discriminator** to encourage device-invariant features via **gradient reversal**.
- Trained with both classification and adversarial domain losses.

### ▶️ Run Training

```bash
python DAR-NN_train.py
```

### 📥 Input Files Required

- `Data_processed/Train/noisy_traces.npy` – Noisy input traces
- `Data_processed/Train/groundtruth_traces.npy` – Clean traces for reconstruction
- `Data_processed/Train/key_labels.npy` – Key labels (0–255 class)
- `Data_processed/Train/domain_labels.npy` – Domain labels (e.g., different devices)

### 💾 Output

- Trains the full model and prints the summary.
- Trained model can be manually saved after training by modifying the script, e.g.:

```python
model.save('DAR-NN_model/DAR-NN_4.h5')
```

---

## 🧪 2. Evaluating the Model (`PGE_test.py`)

### 🔧 Description

- Loads a trained model.
- Computes **Partial Guessing Entropy (PGE)** on side-channel traces from devices D6–D10.
- Repeats testing multiple times with random sampling for statistical significance.

### ▶️ Run Testing

```bash
python PGE_test.py
```

### ⚙️ Configurable Parameters

Modify in `__main__` section:

- `models_folder`: path to trained model
- `file`: name of `.h5` model file
- `num_trace`: number of traces to use per test
- `interest_byte`: byte index of key to attack (usually 0)
- `num_avg`: averaging factor, used in naming convention of test files

### 📥 Input Files Required

Inside each device folder `Data_processed/Test/D[6-10]/`:
- `1_nor_traces_maxmin.npy` – Normalized traces
- `1_10th_roundkey.npy` – True AES key
- `1_ct.npy` – Corresponding ciphertexts

### 📊 Output

- Prints `num_rank` for each device.
- Final print shows **average number of traces required** to correctly identify the key byte:
  
```
Result for D6: 1458
Result for D7: 1283
...
Average num_rank across D6-D10: 1321.4
```

---

## 🛠 Dependencies

Install the following Python packages:

```bash
pip install tensorflow numpy matplotlib tqdm h5py
```

---

## 📌 Notes

- The **Gradient Reversal Layer (GRL)** in training promotes feature invariance across domains.
- The **PGE metric** provides a robust evaluation for side-channel attack models.
- Modify and expand the training/evaluation code as needed for other AES bytes or noise configurations.

---

## 🧑‍💻 Authors & Attribution

If you use this work, please cite or credit the relevant academic research or authorship.



## Database

The first dataset is a far-field EM side-channel dataset, in which the traces are captured at 15 m distance in an office corridor environment from 10 nRF52 DK implementation of TinyAES.
