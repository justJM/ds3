
#In[1]

import tensorflow
from tensorflow.python.client import device_lib
def get_available_device():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU' or 'GPU']

print(get_available_device())


#%%
import tensorflow as tf
def gpu_device():
    with tf.device('/cpu:0'):
        a = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],shape = [2,3],name='a')
        b = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],shape = [3,2],name='b')
    c= tf.matmul(a,b)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess.run([c]))
    sess.close
gpu_device()