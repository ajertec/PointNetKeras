import h5py
import numpy as np
from keras.utils import to_categorical
import keras.backend as K
from keras.regularizers import Regularizer

import tensorflow as tf


##                ##
##  Data Loaders. ##
##                ##

def load_h5(filepath):
    with h5py.File(filepath, "r") as f:
        data = f["data"].value
        label = f["label"].value
    return data, label

def load_h5_seg(h5_filename):
    with h5py.File(h5_filename, "r") as f:
        data = f["data"].value
        label = f["label"].value
        seg = f['pid'][:]
    return data, label, seg

def load_data(train_files):
    """Load all data, used for loading validation data.""" 
    first = True
    for filepath in train_files:
        if first:
            data, label = load_h5(filepath)
            first = False
        else:
            data2, label2 = load_h5(filepath)
            data = np.vstack((data,data2))
            label = np.vstack((label, label2))
            
    return np.expand_dims(data, -1), label
	
def train_generator(train_files, batch_size):

    while True:

      # shuffle list of training .h5 files
      np.random.shuffle(train_files)

      for train_file in train_files:
          
          data, label = load_h5(train_file)
          # shuffle data in every train .h5 file
          data, label, _ = shuffle_data(data, label)

          label = to_categorical(label)

          num_batches = len(label) // batch_size

          # produce batches from train file
          for batch_idx in range(num_batches):
              start_idx = batch_idx * batch_size
              end_idx = (batch_idx+1) * batch_size

              rotated_data = rotate_point_cloud(data[start_idx:end_idx])
              jittered_data = jitter_point_cloud(rotated_data)

              batch_label = label[start_idx:end_idx]

              yield np.expand_dims(jittered_data, -1), batch_label

##                   ##
##  Data Augmenters. ##
##                   ##

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        # Input:
          data: B,N,... numpy array
          label: B,... numpy array
        # Return:
          shuffled data, label and shuffle indices

    url: https://github.com/charlesq34/pointnet/blob/master/provider.py
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        # Input:
          BxNx3 array, original batch of point clouds
        # Return:
          BxNx3 array, rotated batch of point clouds

    url: https://github.com/charlesq34/pointnet/blob/master/provider.py
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        # Input:
          BxNx3 array, original batch of point clouds
        # Return:
          BxNx3 array, jittered batch of point clouds

    url:  https://github.com/charlesq34/pointnet/blob/master/provider.py
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


##                               ##
##  Initializers & Regularizers. ##
##                               ##

def bias_initializer(shape, dtype = None):
    return K.eye(shape, dtype)

def l2_reg(weights):

    """
    Defines the Regularization Function for maintiaing the Orthogonality
    Regularizer Function, Forbenius Norm for ||T(W) * W - I||^2 Norm.

    For Conv2D layer.
    
    url: https://github.com/nbansal90/Can-we-Gain-More-from-Orthogonality/blob/master/ResNet/resnet_cifar_new_100.py

    """
    d_rate = K.variable(0.01)
    w_rate = K.variable(1e-4)

    w = weights
    inp_shape = K.int_shape(w)
    row_dims = inp_shape[0]*inp_shape[1]*inp_shape[2]
    col_dims = inp_shape[3]
    w = K.reshape(w, (row_dims,col_dims))
    W1 = K.transpose(w)
    
    Ident = np.eye(col_dims)
    W_new = K.dot(W1,w)
    Norm  = W_new - Ident

    b_k = np.random.rand(Norm.shape[1])
    b_k = np.reshape(b_k, (Norm.shape[1],1))
    v = K.variable(value=b_k)

    v1 = K.dot(Norm, v)
    norm1 = K.sum(K.square(v1))**0.5

    v2 = tf.divide(v1,norm1)

    v3 = K.dot(Norm,v2)
    return d_rate*(K.sum(K.square(v3))**0.5) + w_rate*(K.sum(K.square(w))**0.5)


def frobenius_norm(w):
    return K.sqrt(K.sum(K.square(K.abs(w))))

def orthogonality_reg(w):

    """
    Orthogonality regularizer for Dense layer based on Frobenius Norm.

    Defines the Regularization Function for maintaining the Orthogonality
    Forbenius Norm for ||T(W) * W - I||^2 Norm

    https://stackoverflow.com/questions/42911671/how-can-i-add-orthogonality-regularization-in-keras
    """

    reg_weight = 0.001

    m = K.dot(K.transpose(w), w) - K.eye(K.int_shape(w)[1])
    return reg_weight * frobenius_norm(m)

  
class OrthoRegularizer(Regularizer):
    """
    Orthogonality regularizer class for Dense layer based on Frobenius Norm.

    Defines the Regularization Function for maintaining the Orthogonality
    Forbenius Norm for ||T(W) * W - I||^2 Norm

    Used when loading pretrained model: Set as custom object.

    https://stackoverflow.com/questions/42911671/how-can-i-add-orthogonality-regularization-in-keras
    """

    def __init__(self, reg_weight = 0.001):
        self.reg_weight = K.cast_to_floatx(reg_weight)

    def __call__(self, w):
        m = K.dot(K.transpose(w), w) - K.eye(K.int_shape(w)[1])
        return self.reg_weight * frobenius_norm(m)


    def get_config(self):
        return {'reg_weight': float(self.reg_weight)}


##                   ##
##     Callbacks.    ##
##                   ##

#TODO
pass