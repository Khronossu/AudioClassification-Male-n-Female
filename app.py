import streamlit as st
import torch
import torchaudio
import torch.nn as nn
import io

# ==========================================
# 1. DEFINE THE MODEL ARCHITECTURE
# ==========================================
# (This MUST match your training code exactly)
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.sigmoid(self.fc(x))

# ==========================================
# 2. LOAD THE TRAINED MODEL
# ==========================================
@st.cache_resource
def load_model():
    model = AudioCNN()
    # Load weights. We use map_location='cpu' so it runs safely on any web server/machine
    model.load_state_dict(torch.load("best_audio_gender_model.pth", map_location=torch.device('cpu'), weights_only=True))
    
    # CRITICAL: Set to evaluation mode! 
    # This turns OFF Dropout and locks Batch Normalization. 
    # If you forget this, your predictions will be random.
    model.eval() 
    return model

model = load_model()

# ==========================================
# 3. AUDIO PREPROCESSING PIPELINE
# ==========================================
def process_audio(audio_bytes):
    # 1. Load audio from memory buffer
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    
    target_sr = 16000
    target_samples = target_sr * 4 # 4 seconds
    
    # 2. Convert to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 3. Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        
    # 4. Pad or Truncate
    if waveform.shape[1] > target_samples:
        waveform = waveform[:, :target_samples]
    elif waveform.shape[1] < target_samples:
        padding = target_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        
    # 5. Extract Mel Spectrogram (Exact same params as training)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=2048, hop_length=512, n_mels=128
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    mel_spec = mel_spectrogram(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    
    # 6. Normalize
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
    
    # Add a "Batch" dimension so the shape becomes [1, 1, 128, time_steps]
    return mel_spec_db.unsqueeze(0) 

# ==========================================
# 4. STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="Voice Gender Classifier", page_icon="🎙️")

st.title("🎙️ Voice Gender Classifier")
st.write("Record a few seconds of your voice, and the CNN will classify it as Male or Female based on your Mel Spectrogram.")

# Streamlit's native microphone widget
audio_value = st.audio_input("Record your voice here:")

if audio_value is not None:
    st.info("Processing audio...")
    
    try:
        # Read the raw bytes from the widget
        audio_bytes = audio_value.read()
        
        # Preprocess into a tensor
        input_tensor = process_audio(audio_bytes)
        
        # Make the prediction
        with torch.no_grad(): # Don't track gradients during inference
            prediction = model(input_tensor)
            prob_female = prediction.item() # Get the float value from the tensor
            
        # Display the results
        st.subheader("Prediction:")
        
        # Remember: 0 = Male, 1 = Female from our training split
        if prob_female > 0.5:
            st.success(f"👧 **Female Voice** (Confidence: {prob_female * 100:.1f}%)")
        else:
            prob_male = 1.0 - prob_female
            st.success(f"👦 **Male Voice** (Confidence: {prob_male * 100:.1f}%)")
            
        # Add a progress bar to visually show the scale
        st.progress(prob_female, text="0% = Male | 100% = Female")
        
    except Exception as e:
        st.error(f"An error occurred while processing the audio: {e}")