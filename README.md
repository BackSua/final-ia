# FractureAI - Sistema de Deteccion de Fracturas Oseas por IA

Sistema de apoyo al diagnostico radiologico basado en inteligencia artificial que analiza imagenes de rayos X para detectar fracturas oseas y generar informes clinicos orientativos.

## Tecnologias

- **Backend:** Django 5.2 + Django REST Framework
- **Modelo:** Red neuronal convolucional ResNet50 (Transfer Learning)
- **Vision por Computadora:** TensorFlow/Keras + OpenCV + Grad-CAM
- **Generacion de informes:** API de Anthropic (Claude)
- **Frontend:** HTML5, CSS3, JavaScript
- **Despliegue:** Render

## Estructura del Proyecto

```
proyecto_ia/
├── backend/                 # Aplicacion Django
│   ├── api/                 # API de prediccion
│   │   ├── views.py         # Endpoint de prediccion + Grad-CAM
│   │   └── ai_report.py     # Generacion de informes con Claude
│   ├── backend/             # Configuracion Django
│   ├── templates/           # Frontend (index.html)
│   └── manage.py
├── training/                # Scripts de entrenamiento
│   └── train_model.py       # Entrenamiento ResNet50
├── modelo/                  # Modelos entrenados y graficas
├── data/                    # Dataset de radiografias
├── requirements.txt
├── Procfile                 # Configuracion Render
├── build.sh                 # Script de build
└── render.yaml              # Configuracion de despliegue
```

## Dataset

**Bone Fracture Binary Classification** de Kaggle:
- ~9,200 imagenes de entrenamiento
- ~830 imagenes de validacion
- ~500 imagenes de prueba
- Clases: `fractured` (fractura) y `not fractured` (normal)

## Instalacion Local

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variable de entorno para informes IA
export ANTHROPIC_API_KEY="tu-api-key"  # Linux/Mac
set ANTHROPIC_API_KEY=tu-api-key       # Windows

# Ejecutar servidor de desarrollo
cd backend
python manage.py migrate
python manage.py runserver
```

Abrir en el navegador: http://127.0.0.1:8000

## Entrenamiento del Modelo

```bash
python training/train_model.py
```

El script genera:
1. `modelo/modelo_fracturas.h5` - Modelo entrenado
2. `modelo/grafica_accuracy_loss.png` - Accuracy y Loss por epoca
3. `modelo/grafica_confusion.png` - Matriz de confusion
4. `modelo/grafica_roc.png` - Curva ROC
5. `modelo/grafica_distribucion.png` - Distribucion de predicciones

## Arquitectura del Modelo

- **Base:** ResNet50 preentrenado en ImageNet
- **Entrenamiento en 2 fases:**
  - Fase 1: Base congelada, entrena solo la cabeza personalizada (5 epocas)
  - Fase 2: Fine-tuning de las ultimas 30 capas (hasta 25 epocas)
- **Cabeza personalizada:** GlobalAveragePooling2D → BatchNorm → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(1, sigmoid)

## Funcionamiento

1. El usuario carga una imagen de rayos X
2. La imagen se preprocesa (resize 224x224, normalizacion ResNet50)
3. El modelo CNN clasifica: fractura o normal
4. Grad-CAM genera un mapa de calor mostrando las regiones de interes
5. La API de Claude analiza la imagen y genera un informe clinico orientativo
6. Se muestra el resultado con el informe, mapa de calor y nivel de confianza

## Despliegue en Render

1. Subir el repositorio a GitHub
2. Crear un nuevo Web Service en Render
3. Conectar el repositorio
4. Configurar las variables de entorno:
   - `ANTHROPIC_API_KEY`: API key de Anthropic
   - `DEBUG`: False
   - `DJANGO_SECRET_KEY`: (se genera automaticamente)
   - `ALLOWED_HOSTS`: .onrender.com
5. Deploy automatico

## Autores

Estudiantes de Ingenieria de Sistemas - CORHUILA
Asignatura: Inteligencia Artificial - 2026-1

## Nota Legal

Este sistema es exclusivamente de apoyo diagnostico. No reemplaza la evaluacion medica profesional. El diagnostico definitivo es responsabilidad del medico especialista.
