import os
import io
import base64
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image, ImageFile
import tensorflow as tf
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Ruta del modelo: relativa al proyecto, portable entre entornos
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    os.path.join(PROJECT_ROOT, 'modelo', 'modelo_fracturas.h5')
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH, compile=False)
        print(f"Modelo cargado: {MODEL_PATH}")
    return _model


def get_gradcam(model, img_array):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
            if 'conv' in layer.name.lower() and hasattr(layer, 'output_shape'):
                if len(layer.output_shape) == 4:
                    last_conv = layer.name
                    break

        if not last_conv:
            return None

        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # predictions[:, 0] = P(fractura) con nuestro mapeo de clases
            # Gradientes de P(fractura) resaltan regiones con fractura
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (224, 224))

        threshold = 0.2
        heatmap = np.where(heatmap >= threshold, heatmap, 0)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        heatmap_rgba = np.zeros((224, 224, 4), dtype=np.uint8)
        heatmap_rgba[:, :, :3] = heatmap_color
        heatmap_rgba[:, :, 3] = np.uint8(heatmap * 180)

        img_pil = Image.fromarray(heatmap_rgba, 'RGBA')
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    except Exception as e:
        print(f"Error Grad-CAM: {e}")
        return None


@csrf_exempt
def predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Solo POST'}, status=405)

    if 'imagen' not in request.FILES:
        return JsonResponse({'error': 'No se recibio imagen'}, status=400)

    try:
        img_file = request.FILES['imagen']
        img = Image.open(img_file).convert('RGB')
        img_resized = img.resize((224, 224))

        img_array = np.array(img_resized).astype('float32')
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch.copy())

        m = get_model()
        pred = float(m.predict(img_preprocessed, verbose=0)[0][0])

        # Mapeo: clase 0 = normal, clase 1 = fractura
        # pred > 0.5 → fractura detectada
        fractura = pred > 0.5
        confianza = pred if fractura else (1.0 - pred)

        gradcam_b64 = get_gradcam(m, img_preprocessed)

        informe = None
        try:
            from .ai_report import generate_report
            buf = io.BytesIO()
            img_resized.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            informe = generate_report(img_b64, fractura, confianza * 100)
        except Exception as e:
            print(f"Error informe IA: {e}")

        return JsonResponse({
            'fractura': fractura,
            'confianza': round(confianza * 100, 2),
            'resultado': 'Fractura detectada' if fractura else 'Sin fractura detectada',
            'prob_fractura': round(pred * 100, 2),
            'prob_normal': round((1.0 - pred) * 100, 2),
            'gradcam': gradcam_b64,
            'informe': informe,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)
