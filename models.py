from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dense, Dropout
from keras.layers import Reshape, Input, dot, concatenate, Lambda
import keras.backend as K

import numpy as np

import utils

def pointnet_cls(number_of_points, number_of_classes):

    """ 
    Classification PointNet. 
    Returns a Keras multi-class (#number_of_classes classes) model.

    # Parameters:
        - number_of_points: integer
            Number of points in pointcloud uniformly sampled from each object.
        - number_of_classes: integer
            Number of object classes.

    Model inputs:   pointcloud, shape (B, number_of_points, 3, 1).
    Model outputs:  class predictions, shape (B, number_of_classes).

    """

    inputs = Input((number_of_points, 3, 1,), name = "point_cloud_input")
    transform_matrix_inp = t_net(inputs, part = "input")
    
    point_cloud_transformed = dot([Reshape((number_of_points, 3))(inputs), transform_matrix_inp], axes = [2,1])
    input_image = Reshape((number_of_points, 3, 1), name = "inp_reshape")(point_cloud_transformed)

    net = Conv2D(64, kernel_size = (1,3), strides=(1, 1), padding='valid', activation = None, name = "conv1")(input_image)
    net = BatchNormalization(name = "conv1_bn")(net)
    net = Activation("relu", name = "conv1_relu")(net)

    net = Conv2D(64, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv2")(net)
    net = BatchNormalization(name = "conv2_bn")(net)
    net = Activation("relu", name = "conv2_relu")(net)

    transform_matrix_feat = t_net(net, part = "feature")
    net_transformed = dot([Reshape((number_of_points, 64))(net), transform_matrix_feat], axes = [2,1])
    net_transformed = Reshape((number_of_points, 1, 64), name = "feat_reshape")(net_transformed)

    net = Conv2D(64, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv3")(net_transformed)
    net = BatchNormalization(name = "conv3_bn")(net)
    net = Activation("relu", name = "conv3_relu")(net)

    net = Conv2D(128, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv4")(net)
    net = BatchNormalization(name = "conv4_bn")(net)
    net = Activation("relu", name = "conv4_relu")(net)

    net = Conv2D(1024, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv5")(net)
    net = BatchNormalization(name = "conv5_bn")(net)
    net = Activation("relu", name = "conv5_relu")(net)

    net = MaxPooling2D(pool_size=(number_of_points, 1), strides=(2,2), padding='valid', name = "maxpool")(net)
    net = Reshape((1024,), name = "maxpool_reshape")(net)

    net = Dense(512, activation = None, name = "fc1")(net)
    net = BatchNormalization(name = "fc1_bn")(net)
    net = Activation("relu", name = "fc1_relu")(net)
    net = Dropout(0.3, name = "dp1")(net)

    net = Dense(256, activation = None, name = "fc2")(net)
    net = BatchNormalization(name = "fc2_bn")(net)
    net = Activation("relu", name = "fc2_relu")(net)
    net = Dropout(0.3, name = "dp2")(net)

    outputs = Dense(number_of_classes, activation="softmax", name = "fc3")(net)

    return Model(inputs = inputs, outputs = outputs)


def pointnet_joint(number_of_points, number_of_classes, number_of_parts):

    """ 
    Joint Classification and Part Segmentation PointNet. 
    Returns a Keras multi-input and multi-output model.

    # Parameters:
        - number_of_points: integer
            Number of points in pointcloud uniformly sampled from each object.
        - number_of_classes: integer
            Number of object classes.
        - number_of_parts: integer
            Number of segmented object parts.

    Model inputs:   pointcloud, shape (B, number_of_points, 3, 1),
                    labels, shape (?????).

    Model outputs:  class predictions, shape (B, number_of_classes),
                    segmentation predictions, shape (B, number_of_points, number_of_parts)

    """

    inputs = Input((number_of_points, 3, 1,), name = "point_cloud_input")

    labels = Input((1,), name = "labels_input")
    labels_one_hot = Lambda(lambda x : K.one_hot(K.cast(x,'int32'), number_of_classes), name = "labels_to_onehot")(labels)

    transform_matrix_inp = t_net(inputs, part = "input")
    
    point_cloud_transformed = dot([Reshape((number_of_points, 3), name = "point_cloud_dropdim")(inputs), transform_matrix_inp], axes = [2,1])
    input_image = Reshape((number_of_points, 3, 1), name = "point_cloud_transf_adddim")(point_cloud_transformed)

    net = Conv2D(64, kernel_size = (1,3), strides=(1, 1), padding='valid', activation = None, name = "conv1")(input_image)
    net = BatchNormalization(name = "conv1_bn")(net)
    out1 = Activation("relu", name = "conv1_relu")(net)

    net = Conv2D(128, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv2")(out1)
    net = BatchNormalization(name = "conv2_bn")(net)
    out2 = Activation("relu", name = "conv2_relu")(net)

    net = Conv2D(128, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv3")(out2)
    net = BatchNormalization(name = "conv3_bn")(net)
    out3 = Activation("relu", name = "conv3_relu")(net)

    transform_matrix_feat = t_net(out3, part = "feature")
    net_transformed = dot([Reshape((number_of_points, 128), name = "feat_dropdim")(out3), transform_matrix_feat], axes = [2,1])
    out_feat = net_transformed = Reshape((number_of_points, 1, 128), name = "feat_transf_adddim")(net_transformed)

    net = Conv2D(512, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv4")(net_transformed)
    net = BatchNormalization(name = "conv4_bn")(net)
    out4 = Activation("relu", name = "conv4_relu")(net)

    net = Conv2D(2048, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv5")(out4)
    net = BatchNormalization(name = "conv5_bn")(net)
    net = Activation("relu", name = "conv5_relu")(net)

    out_max = MaxPooling2D(pool_size=(number_of_points, 1), strides=(2,2), padding='valid', name = "maxpool")(net)

    out_max_rshp = Reshape((2048,), name = "maxpool_dropdims")(out_max)

    # CLASSIFICATION PART
    net = Dense(256, activation = None, name = "cla_fc1")(out_max_rshp)
    net = BatchNormalization(name = "cla_fc1_bn")(net)
    net = Activation("relu", name = "cla_fc1_relu")(net)
    net = Dropout(0.3, name = "cla_dp1")(net) # removed in original part_seg net architecture

    net = Dense(256, activation = None, name = "cla_fc2")(net)
    net = BatchNormalization(name = "cla_fc2_bn")(net)
    net = Activation("relu", name = "cla_fc2_relu")(net)
    net = Dropout(0.3, name = "cla_dp2")(net)

    cla_output = Dense(number_of_classes, activation="softmax", name = "cla_output")(net)


    # SEGMENTATION PART
    labels_one_hot_rshp = Reshape((1, 1, number_of_classes), name = "labels_reshape")(labels_one_hot)
    out_max2 = concatenate([out_max ,labels_one_hot_rshp], axis = -1)

    expand = Lambda(lambda x: K.tile(x, [1, number_of_points, 1, 1]))(out_max2)

    concat = concatenate([out1, out2, out3, out_feat, out4, expand], axis = -1)

    net2 = Conv2D(256, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "seg_conv1")(concat)
    net2 = BatchNormalization(name = "seg_conv1_bn")(net2)
    net2 = Activation("relu", name = "seg_conv1_relu")(net2)
    net2 = Dropout(0.2, name = "seg_dp1")(net2)

    net2 = Conv2D(256, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "seg_conv2")(net2)
    net2 = BatchNormalization(name = "seg_conv2_bn")(net2)
    net2 = Activation("relu", name = "seg_conv2_relu")(net2)
    net2 = Dropout(0.2, name = "seg_dp2")(net2)

    net2 = Conv2D(128, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "seg_conv3")(net2)
    net2 = BatchNormalization(name = "seg_conv3_bn")(net2)
    net2 = Activation("relu", name = "seg_conv3_relu")(net2)

    net2 = Conv2D(number_of_parts, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = "softmax", name = "seg_conv4")(net2)
    seg_output = Reshape((number_of_points, number_of_parts), name = "seg_output")(net2)
 
    return Model(inputs = [inputs, labels], outputs = [cla_output, seg_output])

def pointnet_joint_modified(number_of_points, number_of_classes, number_of_parts):

    """ 
    Modified Joint Classification and Part Segmentation PointNet. 
    Returns a Keras multi-input and multi-output model.
    
    Only difference from pointnet_joint() function is that the labels are removed as input. Model accepts only pointcloud as input. 

    # Parameters:
        - number_of_points: integer
            Number of points in pointcloud uniformly sampled from each object.
        - number_of_classes: integer
            Number of object classes.
        - number_of_parts: integer
            Number of segmented object parts.

    Model inputs:   pointcloud, shape (B, number_of_points, 3, 1),

    Model outputs:  class predictions, shape (B, number_of_classes),
                    segmentation predictions, shape (B, number_of_points, number_of_parts)

    """

    inputs = Input((number_of_points, 3, 1,), name = "point_cloud_input")

    transform_matrix_inp = t_net(inputs, part = "input")
    
    point_cloud_transformed = dot([Reshape((number_of_points, 3), name = "point_cloud_dropdim")(inputs), transform_matrix_inp], axes = [2,1])
    input_image = Reshape((number_of_points, 3, 1), name = "point_cloud_transf_adddim")(point_cloud_transformed)

    net = Conv2D(64, kernel_size = (1,3), strides=(1, 1), padding='valid', activation = None, name = "conv1")(input_image)
    net = BatchNormalization(name = "conv1_bn")(net)
    out1 = Activation("relu", name = "conv1_relu")(net)

    net = Conv2D(128, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv2")(out1)
    net = BatchNormalization(name = "conv2_bn")(net)
    out2 = Activation("relu", name = "conv2_relu")(net)

    net = Conv2D(128, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv3")(out2)
    net = BatchNormalization(name = "conv3_bn")(net)
    out3 = Activation("relu", name = "conv3_relu")(net)

    transform_matrix_feat = t_net(out3, part = "feature")
    net_transformed = dot([Reshape((number_of_points, 128), name = "feat_dropdim")(out3), transform_matrix_feat], axes = [2,1])
    out_feat = net_transformed = Reshape((number_of_points, 1, 128), name = "feat_transf_adddim")(net_transformed)

    net = Conv2D(512, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv4")(net_transformed)
    net = BatchNormalization(name = "conv4_bn")(net)
    out4 = Activation("relu", name = "conv4_relu")(net)

    net = Conv2D(2048, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv5")(out4)
    net = BatchNormalization(name = "conv5_bn")(net)
    net = Activation("relu", name = "conv5_relu")(net)

    out_max = MaxPooling2D(pool_size=(number_of_points, 1), strides=(2,2), padding='valid', name = "maxpool")(net)

    out_max_rshp = Reshape((2048,), name = "maxpool_dropdims")(out_max)

    # CLASSIFICATION PART
    net = Dense(256, activation = None, name = "cla_fc1")(out_max_rshp)
    net = BatchNormalization(name = "cla_fc1_bn")(net)
    net = Activation("relu", name = "cla_fc1_relu")(net)
    net = Dropout(0.3, name = "cla_dp1")(net) # removed in original part_seg net architecture

    net = Dense(256, activation = None, name = "cla_fc2")(net)
    net = BatchNormalization(name = "cla_fc2_bn")(net)
    net = Activation("relu", name = "cla_fc2_relu")(net)
    net = Dropout(0.3, name = "cla_dp2")(net)

    cla_output = Dense(number_of_classes, activation="softmax", name = "cla_output")(net)


    # SEGMENTATION PART
    expand = Lambda(lambda x: K.tile(x, [1, number_of_points, 1, 1]))(out_max)

    # concatenation without onehot encoded labels
    concat = concatenate([out1, out2, out3, out_feat, out4, expand], axis = -1)

    net2 = Conv2D(256, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "seg_conv1")(concat)
    net2 = BatchNormalization(name = "seg_conv1_bn")(net2)
    net2 = Activation("relu", name = "seg_conv1_relu")(net2)
    net2 = Dropout(0.2, name = "seg_dp1")(net2)

    net2 = Conv2D(256, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "seg_conv2")(net2)
    net2 = BatchNormalization(name = "seg_conv2_bn")(net2)
    net2 = Activation("relu", name = "seg_conv2_relu")(net2)
    net2 = Dropout(0.2, name = "seg_dp2")(net2)

    net2 = Conv2D(128, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "seg_conv3")(net2)
    net2 = BatchNormalization(name = "seg_conv3_bn")(net2)
    net2 = Activation("relu", name = "seg_conv3_relu")(net2)

    net2 = Conv2D(number_of_parts, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = "softmax", name = "seg_conv4")(net2)
    seg_output = Reshape((number_of_points, number_of_parts), name = "seg_output")(net2)
 
    return Model(inputs = [inputs], outputs = [cla_output, seg_output])

def pointnet_seg(number_of_points, number_of_parts):

    """ 
    Modified Part Segmentation PointNet. 
    Returns a Keras model.
    

    # Parameters:
        - number_of_points: integer
            Number of points in pointcloud uniformly sampled from each object.
        - number_of_parts: integer
            Number of segmented object parts.

    Model inputs:   pointcloud, shape (B, number_of_points, 3, 1),

    Model outputs:  segmentation predictions, shape (B, number_of_points, number_of_parts)

    """

    inputs = Input((number_of_points, 3, 1,), name = "point_cloud_input")

    transform_matrix_inp = t_net(inputs, part = "input")
    
    point_cloud_transformed = dot([Reshape((number_of_points, 3), name = "point_cloud_dropdim")(inputs), transform_matrix_inp], axes = [2,1])
    input_image = Reshape((number_of_points, 3, 1), name = "point_cloud_transf_adddim")(point_cloud_transformed)

    net = Conv2D(64, kernel_size = (1,3), strides=(1, 1), padding='valid', activation = None, name = "conv1")(input_image)
    net = BatchNormalization(name = "conv1_bn")(net)
    out1 = Activation("relu", name = "conv1_relu")(net)

    net = Conv2D(128, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv2")(out1)
    net = BatchNormalization(name = "conv2_bn")(net)
    out2 = Activation("relu", name = "conv2_relu")(net)

    net = Conv2D(128, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv3")(out2)
    net = BatchNormalization(name = "conv3_bn")(net)
    out3 = Activation("relu", name = "conv3_relu")(net)

    transform_matrix_feat = t_net(out3, part = "feature")
    net_transformed = dot([Reshape((number_of_points, 128), name = "feat_dropdim")(out3), transform_matrix_feat], axes = [2,1])
    out_feat = net_transformed = Reshape((number_of_points, 1, 128), name = "feat_transf_adddim")(net_transformed)

    net = Conv2D(512, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv4")(net_transformed)
    net = BatchNormalization(name = "conv4_bn")(net)
    out4 = Activation("relu", name = "conv4_relu")(net)

    net = Conv2D(2048, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "conv5")(out4)
    net = BatchNormalization(name = "conv5_bn")(net)
    net = Activation("relu", name = "conv5_relu")(net)

    out_max = MaxPooling2D(pool_size=(number_of_points, 1), strides=(2,2), padding='valid', name = "maxpool")(net)

    # SEGMENTATION PART
    expand = Lambda(lambda x: K.tile(x, [1, number_of_points, 1, 1]))(out_max)

    # concatenation without onehot encoded labels
    concat = concatenate([out1, out2, out3, out_feat, out4, expand], axis = -1)

    net2 = Conv2D(256, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "seg_conv1")(concat)
    net2 = BatchNormalization(name = "seg_conv1_bn")(net2)
    net2 = Activation("relu", name = "seg_conv1_relu")(net2)
    net2 = Dropout(0.2, name = "seg_dp1")(net2)

    net2 = Conv2D(256, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "seg_conv2")(net2)
    net2 = BatchNormalization(name = "seg_conv2_bn")(net2)
    net2 = Activation("relu", name = "seg_conv2_relu")(net2)
    net2 = Dropout(0.2, name = "seg_dp2")(net2)

    net2 = Conv2D(128, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = "seg_conv3")(net2)
    net2 = BatchNormalization(name = "seg_conv3_bn")(net2)
    net2 = Activation("relu", name = "seg_conv3_relu")(net2)

    net2 = Conv2D(number_of_parts, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = "softmax", name = "seg_conv4")(net2)
    seg_output = Reshape((number_of_points, number_of_parts), name = "seg_output")(net2)
 
    return Model(inputs = [inputs], outputs = [seg_output])


def t_net(inputs, part):

    """
    Transformation Network, T-Net in paper.

    Keras joint implementation of functions input_transform_net() and feature_transform_net() in: 
    https://github.com/charlesq34/pointnet/blob/master/models/transform_nets.py

    # Parameters:
        inputs : Tensor, 
            - shape = (B, N, 3, 1,) if input transf matrix, 
            - shape = (B, N, 1, n,) if feat transf matrix,
            B - batch size, N - number of points
        part : string, "input" or "feature",
            - input transformation matrix or feature transformation matrix

    # Outputs:
        tranformation matrix, shape (n x n)
        - n = 3 for input transf matrix
        - n = 64 for feature transf matrix
    """
    
    number_of_points = inputs.get_shape().as_list()[1]

    if part not in ["input", "feature"]:
        raise Exception("Parameter '" + str(part)+"' must be either 'input' or 'feature'.")

    if part == "input":
        name_prefix = "inp_"
        n = 3
        net = Conv2D(64, kernel_size = (1,3), strides=(1, 1), padding='valid', activation = None, name = name_prefix+"tconv1")(inputs)
        net = BatchNormalization(name = name_prefix+"tconv1_bn")(net)
        net = Activation("relu", name = name_prefix+"tconv1_relu")(net)
    elif part == "feature": # kernel size changes
        name_prefix = "feat_"
        n = inputs.get_shape().as_list()[-1]
        net = Conv2D(64, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = name_prefix+"tconv1_feat")(inputs)
        net = BatchNormalization(name = name_prefix+"tconv1_bn_feat")(net)
        net = Activation("relu", name = name_prefix+"tconv1_relu_feat")(net)

    # BatchNorm before RELU https://www.youtube.com/watch?v=Xogn6veSyxA&feature=youtu.be&t=325 6min from 5:25
    # https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/

    net = Conv2D(128, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = name_prefix+"tconv2")(net)
    net = BatchNormalization(name = name_prefix+"tconv2_bn")(net)
    net = Activation("relu", name = name_prefix+"tconv2_relu")(net)

    net = Conv2D(1024, kernel_size = (1,1), strides=(1, 1), padding='valid', activation = None, name = name_prefix+"tconv3")(net)
    net = BatchNormalization(name = name_prefix+"tconv3_bn")(net)
    net = Activation("relu", name = name_prefix+"tconv3_relu")(net)

    net = MaxPooling2D(pool_size=(number_of_points, 1), strides=(2,2), padding='valid', name = name_prefix+"tmaxpool")(net)

    net = Reshape((1024,), name = name_prefix+"tmaxpool_rshp")(net)

    net = Dense(512, activation=None, name = name_prefix+"tfc1")(net)
    net = BatchNormalization(name = name_prefix+"tfc1_bn")(net)
    net = Activation("relu", name = name_prefix+"tfc1_relu")(net)

    net = Dense(256, activation=None, name = name_prefix+"tfc2")(net)
    net = BatchNormalization(name = name_prefix+"tfc2_bn")(net)
    net = Activation("relu", name = name_prefix+"tfc2_relu")(net)

    if part == "input":
        kernel_regularizer = None
    elif part == "feature":
        kernel_regularizer = utils.orthogonality_reg

    net = Dense(n*n, activation=None, use_bias=True, weights = [np.zeros((256,n*n)), np.eye(n).flatten()], 
                name = name_prefix+"transform", kernel_regularizer = kernel_regularizer)(net)
    outputs = Reshape((n, n), name = name_prefix+"transform_matrix")(net)

    return outputs