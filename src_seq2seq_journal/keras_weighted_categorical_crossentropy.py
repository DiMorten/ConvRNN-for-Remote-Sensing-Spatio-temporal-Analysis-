"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
from keras import backend as K
import tensorflow as tf
import deb
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def weighted_categorical_crossentropy_ignoring_last_label(weights,
    date_id,t_len):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights_np=weights.copy()
    date_id_np=date_id.copy()
    deb.prints(weights.shape)
    deb.prints(date_id.shape)
    weights = K.variable(weights)
    date_id = K.variable(date_id)
    def loss(y_true, y_pred):
        #date_id=K.zeros_like(y_true)
        #ones_like_y_true=K.ones_like(y_true)


        #date_id=K.switch(K.equal(ones_like_y_true*))
        #date_id=

        # This is a flatten. Flatten y_pred to (batch_size,-1)
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        log_softmax = tf.nn.log_softmax(y_pred)
        
        # Change to one-hot, with class_n+1 channels
        y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1) 
        
        # Eliminate last channel
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1)


        date=0

        deb.prints(K.int_shape(weights))
        deb.prints(K.int_shape(date_id))
        
        deb.prints(K.int_shape(y_true))
        y_true=tf.to_float(y_true)
        deb.prints(K.int_shape(y_true))
        deb.prints(K.int_shape(weights[date]))
        deb.prints(K.int_shape(K.expand_dims(weights[date],axis=0)))
        


        #deb.prints(K.int_shape(K.repeat_elements(
        #   K.expand_dims(K.variable(weights[date]),axis=0),
        #    rep=date_id.shape[0],axis=0)))

        deb.prints(K.int_shape(y_true*K.variable(weights[date])))


        ##deb.prints(K.int_shape(y_true*K.repeat_elements(
        ##    K.expand_dims(K.variable(weights[date]),axis=0)
        ##    ,rep=date_id.shape[0],axis=0)))
       # K.shape(y_true*weights[date])
       # K.shape(lambda: y_true*weights[date])
        
        #cross_entropy = -K.sum(y_true * log_softmax * weights , axis=1)
        cross_entropy = y_true * log_softmax 
        for t_step in range(t_len):
            cross_entropy = K.switch(K.equal(date_id,K.ones_like(date_id)*t_step),
                cross_entropy*K.variable(weights[t_step]), 
                cross_entropy)

        cross_entropy = -K.sum(cross_entropy, axis=1)
        loss = K.mean(cross_entropy)

        return loss
    
    return loss
# 

def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    #y_pred = K.argmax(y_pred,axis=3)
        
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))

import numpy as np
from keras.activations import softmax
from keras.objectives import categorical_crossentropy

# # init tests
# samples=3
# maxlen=4
# vocab=5

# y_pred_n = np.random.random((samples,maxlen,vocab)).astype(K.floatx())
# y_pred = K.variable(y_pred_n)
# y_pred = softmax(y_pred)

# y_true_n = np.random.random((samples,maxlen,vocab)).astype(K.floatx())
# y_true = K.variable(y_true_n)
# y_true = softmax(y_true)

# # test 1 that it works the same as categorical_crossentropy with weights of one
# weights = np.ones(vocab)

# loss_weighted=weighted_categorical_crossentropy(weights)(y_true,y_pred).eval(session=K.get_session())
# loss=categorical_crossentropy(y_true,y_pred).eval(session=K.get_session())
# np.testing.assert_almost_equal(loss_weighted,loss)
# print('OK test1')


# # test 2 that it works differen't than categorical_crossentropy with weights of less than one
# weights = np.array([0.1,0.3,0.5,0.3,0.5])

# loss_weighted=weighted_categorical_crossentropy(weights)(y_true,y_pred).eval(session=K.get_session())
# loss=categorical_crossentropy(y_true,y_pred).eval(session=K.get_session())
# np.testing.assert_array_less(loss_weighted,loss)
# print('OK test2')

# # same keras version as I tested it on?
# import keras
# assert keras.__version__.split('.')[:2]==['2', '0'], 'this was tested on keras 2.0.6 you have %s' % keras.__version
# print('OK version')