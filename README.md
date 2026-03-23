# Voice Gender Classifier

A real-time voice gender classifier built with PyTorch and Streamlit. Record a few seconds of voice in the browser — the app converts it to a Mel Spectrogram and runs it through a trained CNN to predict Male or Female with a confidence score.

## Dataset

Trained on the [Gender Recognition by Voice (Original)](https://www.kaggle.com/datasets/murtadhanajim/gender-recognition-by-voiceoriginal) dataset from Kaggle.

## Engineering Decisions

### 1. Dataset Cleaning — Removing Duplicates via FFT Fingerprinting

The raw dataset contained **1,122 duplicate audio files** that shared identical filenames with different paths. Rather than relying on filename matching (which would miss cross-directory duplicates), I used an **FFT-based audio fingerprint**:

- Load the first 1 second of each file at 16,000 Hz
- Compute the FFT magnitude spectrum
- Hash the first 1,000 frequency bins (rounded to 1 decimal) with MD5

This produced a content-based fingerprint that is fast to compute and robust to metadata differences. All 1,122 duplicates were permanently deleted before any training.

### 2. Dataset Balancing — Strict 1:1 Truncation

The original dataset had more male samples than female. Rather than using oversampling (which can introduce artificial patterns) or weighted loss (which adds a hyperparameter to tune), I used **strict truncation to the minority class count**:

```
Original: 10,380 male files  |  ~5,229 female files
Balanced:  5,229 male files  |   5,229 female files  →  10,458 total
```

This guarantees zero class imbalance with no synthetic data. The dataset is then split 80/20 train/val with `stratify=y` to preserve the 1:1 ratio in both splits.

### 3. Audio Representation — Mel Spectrogram

Raw waveforms are not fed directly to the CNN. Instead, each clip is converted to a **Mel Spectrogram** (in dB scale), which compresses audio into a 2D image where the model can learn frequency patterns with standard Conv2D layers:

| Parameter    | Value  | Reason                                          |
|--------------|--------|-------------------------------------------------|
| `sample_rate`| 16 kHz | All dataset files are natively 16 kHz           |
| `n_fft`      | 2048   | High frequency resolution for pitch differences |
| `hop_length` | 512    | ~32ms frame shift, standard for speech tasks   |
| `n_mels`     | 128    | Covers the full perceptual frequency range      |

After converting to dB, each spectrogram is **z-score normalized** per sample `(x - mean) / (std + 1e-6)` so the CNN receives a consistent input distribution regardless of recording volume.

### 4. Fixed-Length Audio — 4 Seconds

The model requires a fixed input shape. All audio is forced to exactly **4 seconds (64,000 samples at 16 kHz)**:

- **Too long:** Truncate from the end
- **Too short (training):** Zero-pad on the right
- **Too short (inference):** Loop the audio to fill 4 seconds

The loop strategy in inference was added specifically to handle short recordings from Google TTS and browser mic inputs, which zero-padding made sound artificially quiet and biased the model toward predicting Female.

### 5. Model Architecture — Lightweight CNN

A custom 3-block CNN was chosen over a pretrained model (like ResNet) to keep the app lightweight and deployable without a GPU.

```
Input: [1, 128, 125]  ← (channels, mel bins, time frames)

Block 1: Conv2d(1→16,  3×3) → BatchNorm → ReLU → MaxPool(2×2)
Block 2: Conv2d(16→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
Block 3: Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)

Global Average Pool → Dropout(0.4) → Linear(64→1) → Sigmoid
```

**Key choices:**
- **BatchNorm** after every conv layer — stabilizes training and reduces sensitivity to learning rate
- **AdaptiveAvgPool** instead of a flat fully-connected head — makes the model input-size agnostic and reduces parameters significantly
- **Dropout (0.4)** before the classifier — primary regularization to prevent overfitting on the ~8K training samples

### 6. Training Setup

| Setting        | Value               |
|----------------|---------------------|
| Optimizer      | Adam                |
| Learning rate  | 0.001               |
| Loss function  | Binary Cross Entropy|
| Batch size     | 32                  |
| Max epochs     | 50                  |
| Early stopping | patience = 5        |

Early stopping monitors validation loss and saves the best checkpoint (`best_audio_gender_model.pth`). Training stopped at epoch 17 — the model converges quickly due to the clear acoustic separation between male and female pitch ranges.

## How to Run

**1. Create and activate a virtual environment:**

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Confirm the model weights are present:**

Make sure `best_audio_gender_model.pth` is in the same directory as `app.py`.

**4. Launch the app:**

```bash
streamlit run app.py
```
