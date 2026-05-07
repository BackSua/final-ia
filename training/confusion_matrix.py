# Crea un archivo: training\confusion_matrix.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE = r"C:\proyecto_ia\data\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification"
model = load_model(r"C:\proyecto_ia\modelo\modelo_fracturas.h5")

gen = ImageDataGenerator(rescale=1./255)
test_data = gen.flow_from_directory(
    BASE + r"\test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

y_pred = (model.predict(test_data) > 0.5).astype(int).flatten()
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(cm, display_labels=['Fracturado', 'Normal'])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Matriz de Confusión — Test Set', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\proyecto_ia\modelo\grafica_confusion.png', dpi=150)
plt.show()

print("\n📊 REPORTE COMPLETO:")
print(classification_report(y_true, y_pred, target_names=['Fracturado', 'Normal']))