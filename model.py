import tensorflow.keras.layers as layer
import tensorflow as tf
import numpy as np

def make_generator(name="generator") :
    model = tf.keras.Sequential(name=name)
    model.add(layer.Dense(units=7*7*256,kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02)))
    model.add(layer.BatchNormalization())
    model.add(layer.ReLU())
    model.add(layer.Reshape(target_shape=(7,7,256)))
    
    filter_list = [128,64,1]
    stride_list = [1,2,2] 
    for filters,strides in zip(filter_list[:-1],stride_list[:-1]):
        model.add(layer.Conv2DTranspose(filters=filters,strides=strides,kernl_size=(3,3),padding="same",kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02))) 
        model.add(layer.BatchNormalization())
        model.add(layer.ReLU())
        
    model.add(layer.Conv2DTranspose(filters=filter_list[-1],kernel_size=(3,3),strides=stride_list[-1],paddind="same",kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02)))
    
    return model 

def make_discriminator(name="discriminator"):
    model = tf.keras.Sequential(name=name)
    
    filter_list = [64,128,256,512]
    stride_list = [2,2,2,2]
    for filters,strides in zip(filter_list,stride_list):
        model.add(layer.Conv2DTranspose(filters=filters,strides=strides,kernel_size=(3,3),padding="same",kernel_initializer=tf.random_normal_initailizer(mean=0,stddev=0.02)))
        model.add(layer.BatchNormalization())
        model.add(layer.ReLU())
        model.add(layer.Dropout(0.2))
         
    model.add(layer.Flatten())
    model.add(layer.Dense(unit=1,kernel_initializer=tf.random_normal_initializer(mean=0,stdev=0.02)))
    
    return model
         
    