import tensorflow as tf
import tensorflow
from tensorflow import keras
result = tf.config.list_physical_devices('GPU')
print(result)
import numpy as np
import matplotlib.pyplot as plt
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10, 10))
for i in range(25):
  plt.subplot(5, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])
plt.show()
img_inputs = keras.Input(shape=(28, 28))
x0 = tf.keras.layers.Flatten()(img_inputs)
x1 = tf.keras.layers.Dense(128, activation='relu')(x0)
y = tf.keras.layers.Dense(10)(x0)

my_model = keras.Model(inputs=img_inputs, outputs=y, name="my_model")
my_model.summary()
my_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
train_images.shape
train_labels.shape
my_model.fit(train_images, train_labels, epochs=10)



print('Done')
