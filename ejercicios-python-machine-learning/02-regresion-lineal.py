import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
# Cargar datos
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
"""
Tamaño del conjunto de datos y 3 sets sub conjuntos de datos:
training set
test set
validation set
"""
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))
"""
El conjunto de datos se ha cargado con la codificación denominada One-Hot.
Esto significa que las etiquetas se han convertido de un solo número a un vector
 cuya longitud es igual a la cantidad de clases posibles. 
 Todos los elementos del vector son cero excepto el 
 elemento i ésimo que toma el valor uno; y significa que la clase es i.
"""
data.test.labels[0:5, :]