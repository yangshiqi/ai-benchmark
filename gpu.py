import tensorflow as tf

print("Num Nvidia GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num MLUs Available: ", len(tf.config.list_physical_devices('MLU')))
