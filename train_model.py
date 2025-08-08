# ml/train_model.py

import tensorflow as tf
import numpy as np
import os

# Make sure the output folder exists
os.makedirs("ml", exist_ok=True)

print("🔄 Generating dummy patient data...")

# Simulated health data: [age, BMI, BP, glucose]
X = np.random.rand(1000, 4) * [100, 50, 200, 200]
y = np.random.randint(0, 2, size=(1000, 1))

print("🧠 Building and training TensorFlow model...")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Save the model
model_path = "ml/disease_model.h5"
print(f"💾 Saving model to {model_path} ...")
model.save(model_path)

print("✅ Model trained and saved successfully!")
