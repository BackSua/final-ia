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

## Guia Paso a Paso (Windows, nivel principiante)

Esta guia esta escrita para alguien que no ha levantado un proyecto de Python antes.

### Paso 0: Que vas a instalar

- Python 3.10 o 3.11
- Git
- Este repositorio
- Dependencias de Python (se instalan con un comando)

### Paso 1: Instalar Python

1. Ve a [https://www.python.org/downloads/](https://www.python.org/downloads/) y descarga Python 3.10/3.11.
2. Durante la instalacion, marca la casilla **Add Python to PATH**.
3. Termina la instalacion.

### Paso 2: Instalar Git

1. Ve a [https://git-scm.com/download/win](https://git-scm.com/download/win).
2. Instala con opciones por defecto.

### Paso 3: Abrir PowerShell y comprobar que todo esta bien

Ejecuta estos comandos:

```powershell
python --version
pip --version
git --version
```

Si los 3 muestran version, vas bien.

### Paso 4: Descargar el proyecto

En PowerShell, ejecuta:

```powershell
git clone https://github.com/BackSua/final-ia.git
cd final-ia
```

### Paso 5: Crear el entorno virtual (importante)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Si te bloquea PowerShell, ejecuta primero:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

y vuelve a ejecutar `.\venv\Scripts\Activate.ps1`.

### Paso 6: Instalar dependencias

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Este paso puede tardar varios minutos por TensorFlow.

### Paso 7: Configurar variable para informe IA (opcional)

Solo si quieres generar informe con Claude:

```powershell
$env:ANTHROPIC_API_KEY="tu-api-key"
```

Si no configuras esto, la prediccion principal puede funcionar igual.

### Paso 8: Iniciar la aplicacion

```powershell
cd backend
python manage.py migrate
python manage.py runserver
```

Abre en tu navegador:

`http://127.0.0.1:8000`

### Paso 9: Probar que funciona

1. Entra a la pagina.
2. Sube una radiografia.
3. Debes ver clasificacion y porcentaje de confianza.
4. Si agregaste `ANTHROPIC_API_KEY`, tambien veras el informe orientativo.

### Paso 10 (solo si vas a reentrenar): poner la data

Para **usar la app**, no necesitas dataset local.
Para **entrenar de nuevo** si necesitas descargar y ubicar la data.

1. Descarga el dataset "Bone Fracture Binary Classification" de Kaggle.
2. Descomprime el archivo.
3. Copia la carpeta para que quede exactamente en esta ruta:

`C:\proyecto_ia\data\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification\`

4. Verifica que existan estas carpetas:
   - `train\fractured`
   - `train\not fractured`
   - `val\fractured`
   - `val\not fractured`
   - `test\fractured`
   - `test\not fractured`

Si no existe esa estructura, el entrenamiento falla.

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

1. Subir el repositorio a GitHub.
2. Crear un nuevo **Web Service** en Render y conectar el repositorio.
3. Verificar configuracion:
   - Build Command: `./build.sh`
   - Start Command (Procfile): `web: cd backend && gunicorn backend.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
4. Definir variables de entorno en Render:
   - `ANTHROPIC_API_KEY`: API key de Anthropic
   - `DEBUG`: `False`
   - `DJANGO_SECRET_KEY`: una clave segura (larga y aleatoria)
   - `ALLOWED_HOSTS`: `.onrender.com`
   - `CORS_ALLOW_ALL`: `True` o `False` segun politica
5. Desplegar y validar el endpoint publico de la app.

## Autores

Estudiantes de Ingenieria de Sistemas - CORHUILA
Asignatura: Inteligencia Artificial - 2026-1

## Nota Legal

Este sistema es exclusivamente de apoyo diagnostico. No reemplaza la evaluacion medica profesional. El diagnostico definitivo es responsabilidad del medico especialista.
