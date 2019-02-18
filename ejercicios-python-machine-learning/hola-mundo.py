import os
import tensorflow as tf
"""
Deshabilita el warning AVX/FMA
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# esto es tensorflow
hello = tf.constant('Hola Mundo, en hora buena TensorFlow')
sess = tf.Session()
print(sess.run(hello))