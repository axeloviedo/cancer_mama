from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo
model = load_model("modelo_mejorado.keras")

# Tamaño de la imagen de entrada
img_size = 224

# Ruta principal para mostrar el formulario
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para manejar la subida de imagen y la predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Cargar y preprocesar la imagen
        img = Image.open(file).convert('RGB')
        img = img.resize((img_size, img_size))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Para hacerlo compatible con el modelo
        
        # Hacer la predicción
        prediction = model.predict(img_array)
        idc_prob = float(prediction[0][0])  # Probabilidad de IDC
        normal_prob = 1 - idc_prob  # Probabilidad de normal
        
        # Convertir la imagen a Base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Interpretar el resultado
        result = "Carcinoma Ductal Invasivo" if idc_prob > 0.5 else "Normal"

        # Redondear probabilidades
        idc_prob = round(idc_prob, 6)
        normal_prob = round(normal_prob, 6)

        return render_template(
            'index.html',
            prediction=result,
            filename=file.filename,
            idc_prob=idc_prob,
            normal_prob=normal_prob,
            image_data=img_base64
        )

if __name__ == '__main__':
    app.run(debug=True)
