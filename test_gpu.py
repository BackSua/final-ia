import tensorflow as tf
import numpy as np

print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ NumPy: {np.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detectada: {gpus[0]}")
else:
    print("⚠️ Usando CPU — igual funciona")