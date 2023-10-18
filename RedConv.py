import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy
import os
import cv2
from GenerarDatos import IMAGE_SIZE, CATEGORIAS

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

neuronas = [32, 64, 128]
densas = [0, 1, 2]
convpoo = [1, 2, 3]
drop = [0]

x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

x = x / 255.0
y = numpy.array(y)

# Prepara una imagen para que pueda ser utilizada en la red neuronal
def prepare(dir):
    img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

# Predice la etiqueta de una imagen
def predecir():
    #carga el modelo
    pred = tf.keras.models.load_model("models/RedConv-n128-cl2-d2-dropout0")
    #categoriza la imagen
    print(CATEGORIAS[int(pred.predict([prepare('/rex.jpg')])[0][0])])

# Entrena red neuronal convolucional
def entrenar():
    # Recorre todas las posibles combinaciones de parámetros
    for neurona in neuronas:
        for conv in convpoo:
            for densa in densas:
                for d in drop:

                    # Crea un nombre único para el modelo
                    NAME = "RedConv-n{}-cl{}-d{}-dropout{}".format(neurona,conv,densa,d)

                    # Crea un callback para guardar las métricas del entrenamiento
                    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

                    # Crea una red neuronal convolucional
                    model = Sequential()
                    model.add(Conv2D(64, (3,3), input_shape = x.shape[1:]))
                    model.add(Activation("relu"))
                    model.add(MaxPooling2D(pool_size = (2,2)))

                    # Si se ha activado el dropout, agrega una capa de dropout
                    if d == 1:
                        model.add(Dropout(0.2))

                    # Agrega capas convolucionales y de max pooling
                    for i in range(conv):
                        model.add(Conv2D(64, (3,3)))
                        model.add(Activation("relu"))
                        model.add(MaxPooling2D(pool_size = (2,2)))

                    # Agrega una capa flatten para aplanar la salida de las capas convolucionales
                    model.add(Flatten())

                    # Agrega capas densas
                    for i in range(densa):
                        model.add(Dense(neurona))
                        model.add(Activation("relu"))

                    # Agrega una capa densa final para predecir la etiqueta
                    model.add(Dense(1))
                    model.add(Activation('sigmoid'))

                    # Compila la red neuronal
                    model.compile(loss="binary_crossentropy",
                                  optimizer="adam",
                                  metrics=['accuracy'])

                    # Entrena la red neuronal
                    model.fit(x,y, batch_size = 30, epochs = 10, validation_split = 0.3, callbacks=[tensorboard])

                    # Guarda la red neuronal entrenada
                    model.save("models/{}".format(NAME))

# Predice la etiqueta de la imagen 'rex.jpg'
predecir()
