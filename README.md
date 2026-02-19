# Voice Gender Classifier

## Dataset
This project was trained using the [Gender Recognition by Voice (Original)](https://www.kaggle.com/datasets/murtadhanajim/gender-recognition-by-voiceoriginal) dataset from Kaggle.

## How to Run the Project

Follow these steps to set up and run the Streamlit application on your local machine:

**1. Create and activate a Python virtual environment:**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**2. Install the required dependencies:**

```bash
pip install -r requirements.txt
```

**3. Prepare the model weights:**

Make sure your trained model file (e.g., `best_audio_gender_model.pth`) is located in the exact same directory as your `app.py` script.

**4. Launch the Streamlit app:**

```bash
streamlit run app.py
```

