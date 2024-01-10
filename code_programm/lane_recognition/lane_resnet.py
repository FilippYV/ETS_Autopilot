import tensorflow as tf


print(tf.config.list_physical_devices())
resnet = tf.keras.applications.resnet.ResNet101