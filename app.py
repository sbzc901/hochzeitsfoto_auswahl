import streamlit as st
import os
import cv2
import zipfile
import shutil
import numpy as np
import face_recognition
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, as_completed

# Konfiguration
st.set_page_config(page_title="Hochzeitsfoto-Auswahl", layout="wide")
st.title("🤵👰 Automatische Auswahl der besten Hochzeitsfotos")

MAX_FILES = 4000
TOP_N = st.slider("Wie viele Top-Fotos möchtest du auswählen?", 10, 1000, 100)

# Bildbewertung

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 100

def process_image(path):
    try:
        image = cv2.imread(path)
        if image is None:
            return path, 0
        
        score = 0
        
        # Blur-Check
        if not is_blurry(image):
            score += 1
        
        # DeepFace Emotion Analysis
        result = DeepFace.analyze(img_path=path, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        emotion = result.get('dominant_emotion', '')
        if emotion in ['happy', 'surprise']:
            score += 2
        else:
            score += 1
            
        return path, score
    except:
        return path, 0

# Datei-Upload
uploaded_files = st.file_uploader(
    "Lade bis zu 4000 Bilder hoch", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > MAX_FILES:
        st.warning(f"Bitte nicht mehr als {MAX_FILES} Dateien auf einmal hochladen.")
    else:
        with TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            st.write(f"⏳ Verarbeite {len(uploaded_files)} Bilder...")
            
            image_paths = []
            for file in uploaded_files:
                path = os.path.join(input_dir, file.name)
                with open(path, "wb") as f:
                    f.write(file.read())
                image_paths.append(path)
            
            results = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(process_image, p) for p in image_paths]
                progress_bar = st.progress(0)
                for i, f in enumerate(as_completed(futures)):
                    results.append(f.result())
                    progress_bar.progress((i + 1) / len(futures))
            
            # Diese Zeilen stehen NACH dem ThreadPoolExecutor (12 Leerzeichen)
            results.sort(key=lambda x: x[1], reverse=True)
            top_paths = [r[0] for r in results[:TOP_N]]
            for path in top_paths:
                shutil.copy(path, os.path.join(output_dir, os.path.basename(path)))
            
            # Anzeige der besten Bilder
            st.success(f"🎉 Fertig! Die besten {TOP_N} Bilder wurden ausgewählt.")
            cols = st.columns(5)
            for i, path in enumerate(top_paths[:20]):
                with cols[i % 5]:
                    st.image(path, use_column_width=True)
            
            # ZIP-Datei erstellen
            zip_path = os.path.join(temp_dir, "top_bilder.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for file in os.listdir(output_dir):
                    zipf.write(os.path.join(output_dir, file), arcname=file)
            
            with open(zip_path, "rb") as f:
                st.download_button("📦 ZIP-Datei herunterladen", f, file_name="Top_Bilder.zip")
