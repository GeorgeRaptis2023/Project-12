import os,PIL,PIL.Image
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

train_ds = tf.keras.utils.image_dataset_from_directory('C:/Users/argra/Desktop/repos/Project-12/train',validation_split=0.4,subset="training",seed=123,image_size=(200, 200), batch_size=2)
val_ds = tf.keras.utils.image_dataset_from_directory('C:/Users/argra/Desktop/repos/Project-12/train', validation_split=0.4,subset="validation",seed=123,image_size=(200, 200), batch_size=2)
class_names=train_ds.class_names

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
image_batch, labels_batch = next(iter(train_ds.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y))))
model = tf.keras.Sequential([tf.keras.layers.Rescaling(1./255),tf.keras.layers.Conv2D(32, 3, activation='relu'),tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),tf.keras.layers.Flatten(),tf.keras.layers.Dense(128, activation='relu'),tf.keras.layers.Dense(3)
])
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(train_ds,validation_data=val_ds,epochs=8)
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])


img_path = '1704312584184.jpg'
img = image.load_img(img_path, target_size=(200, 200))

img = image.img_to_array(img)

img = preprocess_input(img, data_format=None)
img = img/255
img = np.expand_dims(img, axis=0)

print(class_names[np.argmax(probability_model.predict(img)[0])])