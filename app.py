import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Modeli Yükleme
model = tf.keras.models.load_model('model.h5')  # Modelinizi kaydederken 'model.h5' adıyla kaydedin

# Başlık
st.title('Röntgen Görüntü Sınıflandırma Uygulaması')

# Kullanıcıdan Resim Yükleme
st.subheader('Röntgen Görüntüsünü Yükleyin')
uploaded_file = st.file_uploader("Bir resim yükleyin", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Yüklenen Resmi Okuma ve Görüntüleme
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Röntgen Görüntüsü', use_column_width=True)
    st.write("Yükleniyor...")

    # Eğer görüntü RGB değilse, tek kanal haline getirme (örneğin gri tonlama)
    if image.mode != 'RGB':
        image = image.convert('RGB')  # RGB'ye dönüştürülüyor

    # Resmi Model için Hazırlama
    image = image.resize((224, 224))  # Modelin giriş boyutu ile uyumlu hale getirme
    image = np.array(image) / 255.0  # Normalize etme
    image = np.expand_dims(image, axis=0)  # Batch boyutunu ekle

    # Tahmin Yapma
    prediction = model.predict(image)
    if prediction[0] > 0.5:
        st.write("Tahmin: Pozitif Sınıf (örn. Anormal)")  # Pozitif sınıf adı
    else:
        st.write("Tahmin: Negatif Sınıf (örn. Normal)")  # Negatif sınıf adı
