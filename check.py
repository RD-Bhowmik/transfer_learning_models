import tensorflow as tf

if tf.test.gpu_device_name():
    print("Default GPU Device:", tf.test.gpu_device_name())
else:
    print("Please install GPU version of TF, or check your CUDA and cuDNN setup.")
