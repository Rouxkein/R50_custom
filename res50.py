import numpy as np
import tensorflow as tf
import keras

import tensorflow.python.keras.callbacks
from tensorflow.keras.layers import Conv2D,Input,BatchNormalization,\
    ZeroPadding2D,Activation,MaxPool2D,Add,AveragePooling2D,Flatten,Dense
from keras.models import Model,load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import glorot_uniform

data_dir="/Downloads/archive/Chessman-image-dataset/Chess"
train_ds=image_dataset_from_directory(directory=data_dir,batch_size=32,
                                      image_size=(224,224),subset="trainning",
                                      validation_split=0.2)
validation_ds=image_dataset_from_directory(directory=data_dir,
                                           batch_size=32,image_size=(224,224),
                                           subset="validation",validation_split=0.2)
# Resnet50 (identity block ,convolution block,Grade Funtion

def Identity_block(X,f,filters,stage,block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    #in this block input(x)-->conv2d=>BN=>relu=>Conv2d=>BN=>input(x)=>relu
    conv_name_base='res'+str(stage)+block+'_brach'
    bn_name_base='bn'+str(stage)+block+'_brach'
    F1,F2,F3=filters
    X_shortcut=X

    X=Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),
             padding='valid',kernel_initializer=glorot_uniform(seed=0),
             name=conv_name_base+'2a')(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2a')(X)
    X=Activation('relu')(X)

    X=Conv2D(filters=F2,kernel_size=(1,1),strides=(1,1),
             padding='valid',kernel_initializer=glorot_uniform(seed=0),
             name=conv_name_base+'2b')(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X=Activation('relu')(X)

    X=Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding='valid',
             kernel_initializer=glorot_uniform(seed=0),
             name=conv_name_base+'2c')(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2c')

    X=Add()[(X_shortcut,X)]
    X=Activation('relu')(X)
    return X
def ConvolutionBlock(X,f,filters,stage,block,s=2):
    cv_name_base='res'+str(stage)+block+'_brach'
    bn_name_base='bn' + str(stage)+block+'_brach'
    F1,F2,F3 = filters
    X_shortcut=X

    X=Conv2D(filters=F1,
             kernel_size=(1,1),strides=(s,s),
             padding='valid',
             kernel_initializer=glorot_uniform(seed=0)
             ,name=cv_name_base+'2a')(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2a')
    X=Activation('relu')(X)

    X = Conv2D(filters=F2,
               kernel_size=(1, 1), strides=(s, s),
               padding='valid',
               kernel_initializer=glorot_uniform(seed=0)
               , name=cv_name_base + '2b')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')
    X = Activation('relu')(X)

    X =Conv2D(filters=F3,
              kernel_size=(1,1),strides=(s,s),
              padding='valid',
              kernel_initializer=glorot_uniform(seed=0),name=cv_name_base+'2c')(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2c')(X)

    X_shortcut=Conv2D(filters=F3,
              kernel_size=(1,1),strides=(s,s),
              padding='valid',
              kernel_initializer=glorot_uniform(seed=0),name=cv_name_base+'1')(X)
    X_shortcut=BatchNormalization(axis=3,name=bn_name_base+'1')
    X=Add([X,X_shortcut])
    X=Activation('relu')(X)
    return X

def R50(image_size=(224,224,3),classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    X_input = Input(image_size)
    X=ZeroPadding2D((3,3))(X_input)
    # stage 1
    X=Conv2D(64,(7,7),strides=(2,2),
             name='conv1',kernel_initializer = glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name='bn_conv1')(X)
    X=Activation('relu')(X)
    X=MaxPool2D((3,3),strides=(2,2))(X)

    #stage 2
    X=ConvolutionBlock(X,3,filters=[64,64,256],stage=3,block='a',s=1)
    X=Identity_block(X,3,[64,64,256],stage=2,block='b')
    X=Identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    #stage 3
    X = ConvolutionBlock(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = Identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = Identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = Identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    #stage 4
    X = ConvolutionBlock(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = Identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = Identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = Identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = Identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = Identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    # Stage 5 (≈3 lines)
    X = X = ConvolutionBlock(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = Identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = Identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model
model=R50(image_size=(224,224,3),classes=6)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
my_callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(train_ds,epochs=10,validation_data=validation_ds,batch_size=32,callbacks=my_callbacks)
model.save('res_chess')
