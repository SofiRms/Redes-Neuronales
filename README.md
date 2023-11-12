# Redes-Neuronales
Proyecto de Python y TensorFlow, uso de red neuronal convolucional - Modelo para clasificación de imágenes - Categorías "perro" y "gato" 

![TensorFlow-Python](/images/TensorFlow-Python.png)


## Como usar:
Descargar el codigo haciendo uso de `git clone https://github.com/SofiRms/Redes-Neuronales.git`

## Librerias:
1. Acceder al directorio del proyecto
2. Ubicar el dataset de imagenes en el directorio raíz
3. Ejecutar comando para instalación de dependencias
 ```shell
  `pip install -r requirements.txt`
```
5. Reemplazar directorio de imagen para predicción, en DATADIR de archivo GenerarDatos.py, con imagen deseada
 

6. Ejecutar comando para generación de datos de entrenamiento y testing
 ```shell
python GenerarDatos.py
```
5. Ejecutar modelo
 ```shell
python RedConv.py
```
