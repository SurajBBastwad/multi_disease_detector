import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle

# Optional: Force CPU if GPU causes issues
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Paths
X_DIR = "model/X_chunks"
Y_DIR = "model/y_chunks"
MODEL_PATH = "model/cnn_model.h5"

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32  # Reduced for stability
EPOCHS = 10
NUM_CLASSES = 6

# Load chunked data
def load_data():
    X = []
    y = []
    files = sorted(os.listdir(X_DIR))
    for fname in files:
        x = np.load(os.path.join(X_DIR, fname)).astype(np.float32)
        yname = fname.replace("img_", "label_")
        yval = int(np.load(os.path.join(Y_DIR, yname)))
        X.append(x)
        y.append(yval)
    X = np.array(X, dtype=np.float32)
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)
    X, y = shuffle(X, y, random_state=42)
    return X, y

# Build CNN model
def build_model():
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and save
X, y = load_data()
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

model = build_model()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Plot training history
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("model/training_plot.png")
plt.show()

# Save model
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
