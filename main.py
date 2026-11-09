import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import speech_recognition as sr
import gradio as gr

# 1. Baca dataset
df = pd.read_csv("dataset_percakapan_sunda.csv")

# Pastikan tidak ada data yang kosong
df = df.dropna()

# 2. Pisahkan teks dan label
X = df['teks_ucapan']
y = df['label_tutur']

# 3. Bagi data menjadi 80% untuk pelatihan (training) dan 20% untuk pengujian (testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Ubah teks menjadi vektor angka menggunakan TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Data pelatihan siap: {X_train_vec.shape[0]} baris")

# --- Model 1: Naive Bayes ---
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
nb_akurasi = accuracy_score(y_test, nb_pred)

# --- Model 2: Support Vector Machine (SVM) ---
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_vec, y_train)
svm_pred = svm_model.predict(X_test_vec)
svm_akurasi = accuracy_score(y_test, svm_pred)

print(f"Akurasi Naive Bayes: {nb_akurasi * 100:.2f}%")
print(f"Akurasi SVM: {svm_akurasi * 100:.2f}%")

# Lihat detail performa SVM (Presisi, Recall, F1-Score)
print("\nLaporan Detail Klasifikasi SVM:")
print(classification_report(y_test, svm_pred))

def deteksi_suara_sunda(audio_path):
    if audio_path is None:
        return "Teu aya sora (Tidak ada suara)."
        
    recognizer = sr.Recognizer()
    try:
        # Membaca file audio dari mikrofon Gradio
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            
        # Transkripsi menggunakan Google Speech API (id-ID cukup baik menangkap kosakata Sunda)
        teks_transkripsi = recognizer.recognize_google(audio_data, language="id-ID")
        
        # Eksekusi model NLP
        vektor_input = vectorizer.transform([teks_transkripsi.lower()])
        prediksi = svm_model.predict(vektor_input)[0]
        
        return f"Teks nu kadangu: '{teks_transkripsi}'\n\nHasil Analisis: SUNDA {prediksi.upper()}"
        
    except sr.UnknownValueError:
        return "Sora teu kadangu bener (Suara tidak terdengar jelas). Coba deui."
    except sr.RequestError:
        return "Gangguan koneksi API API Speech-to-Text."
    except Exception as e:
        return f"Error sistem: {str(e)}"

# Buat UI dengan input Audio
ui_suara = gr.Interface(
    fn=deteksi_suara_sunda,
    inputs=gr.Audio(type="filepath", label="Rekam Sora Anjeun (Rekam Suaramu)"),
    outputs=gr.Text(label="Hasil Deteksi"),
    title="🎙️ Pendeteksi Suara Sunda Lemes & Kasar",
    description="Pencet tombol rekam, ucapkeun kalimat basa Sunda, sistem bakal nganalisis tingkat tuturna."
)

ui_suara.launch()