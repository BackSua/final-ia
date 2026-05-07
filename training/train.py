import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Configuración ──
BASE = r"C:\proyecto_ia\data\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification"
TRAIN_DIR = os.path.join(BASE, "train")
VAL_DIR   = os.path.join(BASE, "val")
TEST_DIR  = os.path.join(BASE, "test")
IMG_SIZE  = 224
BATCH     = 32
EPOCHS    = 20

# ── Datos ──
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2]
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode='binary'
)
val_data = val_gen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode='binary'
)
test_data = val_gen.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode='binary', shuffle=False
)

print(f"✅ Train: {train_data.samples} imágenes")
print(f"✅ Val:   {val_data.samples} imágenes")
print(f"✅ Test:  {test_data.samples} imágenes")
print(f"✅ Clases: {train_data.class_indices}")

# ── Modelo FASE 1 — base congelada ──
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

print("\n🔒 FASE 1 — Base congelada (5 épocas)...")
history1 = model.fit(train_data, validation_data=val_data, epochs=5)

# ── FASE 2 — Fine-tuning ──
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(0.00005),
              loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint(r'C:\proyecto_ia\modelo\mejor_modelo.h5',
                    monitor='val_accuracy', save_best_only=True)
]

print("\n🔓 FASE 2 — Fine-tuning (hasta 20 épocas)...")
history2 = model.fit(train_data, validation_data=val_data,
                     epochs=EPOCHS, callbacks=callbacks)

# ── Evaluación final ──
loss, acc = model.evaluate(test_data)
print(f"\n🏆 Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")

# ── Guardar modelo final ──
model.save(r'C:\proyecto_ia\modelo\modelo_fracturas.h5')
print("✅ Modelo guardado en C:\\proyecto_ia\\modelo\\")

# ── Gráficas ──
acc_total = history1.history['accuracy'] + history2.history['accuracy']
val_acc_total = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss_total = history1.history['loss'] + history2.history['loss']
val_loss_total = history1.history['val_loss'] + history2.history['val_loss']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, len(acc_total) + 1)

ax1.plot(epochs_range, acc_total, 'b-o', label='Entrenamiento')
ax1.plot(epochs_range, val_acc_total, 'r-o', label='Validación')
ax1.set_title('Accuracy por Época', fontweight='bold')
ax1.set_xlabel('Época')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs_range, loss_total, 'b-o', label='Entrenamiento')
ax2.plot(epochs_range, val_loss_total, 'r-o', label='Validación')
ax2.set_title('Loss por Época', fontweight='bold')
ax2.set_xlabel('Época')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(r'C:\proyecto_ia\modelo\grafica_accuracy_loss.png', dpi=150)
plt.show()
print("✅ Gráficas guardadas")