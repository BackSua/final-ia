import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from PIL import ImageFile
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Configuracion ──
BASE = r"C:\proyecto_ia\data\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification"
TRAIN_DIR = os.path.join(BASE, "train")
VAL_DIR = os.path.join(BASE, "val")
TEST_DIR = os.path.join(BASE, "test")
OUTPUT_DIR = r"C:\proyecto_ia\modelo"
IMG_SIZE = 224
BATCH = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Generadores de datos ──
# CRITICO: classes=['not fractured', 'fractured'] fuerza el mapeo:
#   Clase 0 = "not fractured" (normal)
#   Clase 1 = "fractured" (fractura)
# Asi pred > 0.5 siempre significa FRACTURA DETECTADA

CLASS_ORDER = ['not fractured', 'fractured']

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    fill_mode='nearest'
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode='binary',
    classes=CLASS_ORDER
)

val_data = val_gen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode='binary',
    classes=CLASS_ORDER
)

test_data = val_gen.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode='binary',
    classes=CLASS_ORDER, shuffle=False
)

print(f"\n{'='*50}")
print(f"MAPEO DE CLASES: {train_data.class_indices}")
print(f"  Clase 0 = Normal (not fractured)")
print(f"  Clase 1 = Fractura (fractured)")
print(f"  Train: {train_data.samples} imagenes")
print(f"  Val:   {val_data.samples} imagenes")
print(f"  Test:  {test_data.samples} imagenes")
print(f"{'='*50}\n")

with open(os.path.join(OUTPUT_DIR, 'class_indices.json'), 'w') as f:
    json.dump(train_data.class_indices, f, indent=2)

# ── Modelo: ResNet50 + cabeza personalizada ──
base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base.input, outputs=output)

# ── FASE 1: Base congelada (5 epocas) ──
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("FASE 1 - Base congelada (5 epocas)...")
h1 = model.fit(train_data, validation_data=val_data, epochs=5)

# ── FASE 2: Fine-tuning ultimas 30 capas ──
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'mejor_modelo.h5'),
        monitor='val_accuracy', save_best_only=True
    )
]

print("\nFASE 2 - Fine-tuning (hasta 25 epocas)...")
h2 = model.fit(train_data, validation_data=val_data, epochs=25, callbacks=callbacks)

# ── Evaluacion final ──
loss, acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.1f}%)")
print(f"Test Loss: {loss:.4f}")

model.save(os.path.join(OUTPUT_DIR, 'modelo_fracturas.h5'))
print(f"Modelo guardado en {OUTPUT_DIR}")

# ═══════════════════════════════════════
# GRAFICAS (minimo 3 requeridas)
# ═══════════════════════════════════════

acc_total = h1.history['accuracy'] + h2.history['accuracy']
val_acc_total = h1.history['val_accuracy'] + h2.history['val_accuracy']
loss_total = h1.history['loss'] + h2.history['loss']
val_loss_total = h1.history['val_loss'] + h2.history['val_loss']
epochs_range = range(1, len(acc_total) + 1)
fase1_end = len(h1.history['accuracy'])

# ── Grafica 1: Accuracy y Loss ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_range, acc_total, 'b-o', label='Entrenamiento', markersize=4)
ax1.plot(epochs_range, val_acc_total, 'r-o', label='Validacion', markersize=4)
ax1.axvline(x=fase1_end + 0.5, color='gray', linestyle='--', alpha=0.5, label='Inicio fine-tuning')
ax1.set_title('Accuracy por Epoca', fontweight='bold', fontsize=13)
ax1.set_xlabel('Epoca')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, loss_total, 'b-o', label='Entrenamiento', markersize=4)
ax2.plot(epochs_range, val_loss_total, 'r-o', label='Validacion', markersize=4)
ax2.axvline(x=fase1_end + 0.5, color='gray', linestyle='--', alpha=0.5, label='Inicio fine-tuning')
ax2.set_title('Loss por Epoca', fontweight='bold', fontsize=13)
ax2.set_xlabel('Epoca')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'grafica_accuracy_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Grafica 1: Accuracy y Loss guardada")

# ── Grafica 2: Matriz de Confusion ──
y_pred_proba = model.predict(test_data).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Fractura'])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Matriz de Confusion - Test Set', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'grafica_confusion.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Grafica 2: Matriz de Confusion guardada")

# ── Grafica 3: Curva ROC ──
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Clasificador aleatorio')
ax.fill_between(fpr, tpr, alpha=0.1, color='blue')
ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
ax.set_title('Curva ROC - Deteccion de Fracturas', fontweight='bold', fontsize=13)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'grafica_roc.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Grafica 3: Curva ROC guardada")

# ── Grafica 4: Distribucion de probabilidades ──
fig, ax = plt.subplots(figsize=(8, 5))
pred_normal = y_pred_proba[y_true == 0]
pred_fractura = y_pred_proba[y_true == 1]
ax.hist(pred_normal, bins=30, alpha=0.6, color='green', label='Normal (real)', density=True)
ax.hist(pred_fractura, bins=30, alpha=0.6, color='red', label='Fractura (real)', density=True)
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Umbral (0.5)')
ax.set_xlabel('Probabilidad predicha de fractura')
ax.set_ylabel('Densidad')
ax.set_title('Distribucion de Predicciones', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'grafica_distribucion.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Grafica 4: Distribucion de predicciones guardada")

# ── Reporte de clasificacion ──
print(f"\n{'='*50}")
print("REPORTE DE CLASIFICACION")
print(f"{'='*50}")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Fractura']))
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"\nTodas las graficas guardadas en: {OUTPUT_DIR}")
