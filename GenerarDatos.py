# Importación de bibliotecas necesarias
import os
import cv2
from tqdm import tqdm
import random as rn
import numpy as np
import pickle

# Definición de categorías de imágenes
CATEGORIAS = ["Cat", "Dog"]

# Definición el tamaño de las imágenes
IMAGE_SIZE = 100

# Definición la función para generar el conjunto de datos
def Generar_datos():
    # Crear una lista vacía para almacenar los datos
    data = []

    # Recorrer todas las categorías de imágenes
    for categoria in CATEGORIAS:

        # Obtener la ruta a la carpeta de la categoría actual
        path = os.path.join(DATADIR, categoria)

        # Obtener el índice de la categoría actual
        valor = CATEGORIAS.index(categoria)

        # Obtener una lista de todos los archivos en la carpeta de la categoría actual
        listdir = os.listdir(path)

        # Recorrer todos los archivos en la carpeta de la categoría actual
        for i in tqdm(range(len(listdir)), desc = categoria):

            # Obtener el nombre del archivo actual
            imagen_nombre = listdir[i]

            # Intentar leer el archivo actual
            try:

                # Obtener la ruta completa al archivo actual
                imagen_ruta = os.path.join(path, imagen_nombre)

                # Leer el archivo actual como una imagen en escala de grises
                imagen = cv2.imread(imagen_ruta, cv2.IMREAD_GRAYSCALE)

                # Redimensionar la imagen al tamaño definido
                imagen = cv2.resize(imagen,(IMAGE_SIZE, IMAGE_SIZE))

                # Agregar la imagen y su etiqueta a la lista de datos
                data.append([imagen, valor])

            # Si no se puede leer el archivo, ignorar
            except Exception as e:
                pass

    # Mezclar la lista de datos
    rn.shuffle(data)

    # Crear dos listas vacías para almacenar las imágenes y las etiquetas
    x = []
    y = []

    # Recorrer la lista de datos y agregamos las imágenes y etiquetas a las listas correspondientes
    for i in tqdm(range(len(data)),desc="Procesamiento"):
        par = data[i]
        x.append(par[0])
        y.append(par[1])

    # Convertir la lista de imágenes a un array de NumPy
    x = np.array(x)

    # Dar forma al array de imágenes para que tenga el tamaño correcto para ser utilizado en una red neuronal
    x = x.reshape(-1,IMAGE_SIZE, IMAGE_SIZE,1)

    # Guardar el array de imágenes en un archivo pickle
    pickle_out = open("x.pickle","wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()
    print("Archivo x.pickle creado!")

    # Guardar el array de etiquetas en un archivo pickle
    pickle_out = open("y.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
    print("Archivo y.pickle creado!")

# Definir la ruta al directorio con las imágenes
DATADIR = "C:\\Users\\Usuario\\Desktop\\Redes-Redes-Neuronales\\Images"

# Llamar a la función para generar el conjunto de datos
if __name__ == "__main__":
    Generar_datos()
