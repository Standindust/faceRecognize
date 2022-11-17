import tensorflow.compat.v1 as tf
tf.config.list_physical_devices() # gpu版的安装的信息
#检查在本机有没有安装cuda  cudnn
tf.test.is_built_with_cuda()
tf.test.is_built_with_gpu_support()

