import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Cargar los datos de entrenamiento y prueba
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocesar los datos
X_train = X_train.reshape(-1, 28 * 28) / 255.0  # Aplanar y normalizar las imágenes de entrenamiento
X_test = X_test.reshape(-1, 28 * 28) / 255.0    # Aplanar y normalizar las imágenes de prueba
y_train = to_categorical(y_train)  # Codificar las etiquetas de entrenamiento en one-hot encoding
y_test = to_categorical(y_test)    # Codificar las etiquetas de prueba en one-hot encoding

# Crear el modelo de la red neuronal
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
