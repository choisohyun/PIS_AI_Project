'''
1. functions
    (1). model compile
    (2). prediction
2. dependency: keras, tensorflow, opencv

'''


import keras
from keras.models import model_from_json
import random
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense,Conv3D,Input,MaxPool3D,Activation, GlobalAveragePooling3D, ZeroPadding3D
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import cv2
import os
from keras import backend as K


#   훈련된 모델 불러와서 컴파일 된 모델 반환
def modelCompile():
    json_file = open('/home/pirl/Downloads/real/C3Dmodel/model(lr=0.0001,ADAM,binary,epoch10).json','r')
    model = json_file.read()
    json_file.close()
    model = model_from_json(model)
    model.summary()

    model.load_weights('/home/pirl/Downloads/real/C3Dmodel/weights_c3d(lr=0.0001,ADAM,binary,epoch10).h5',
                       by_name=True)


    lr = 0.0001
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print("c3d model compile is done!")


    model._make_predict_function()



    return model


#   버퍼에서 (1, 128, 171, 16, 3)짜리 프레임 1개 읽어와서 예측 값 반환
def violencePred(model, input):
    pred = model.predict(input)
    if(pred[0][0]<pred[0][1]):# violence detected!
        print("violence detected!!!!!violence detected!!!!!violence detected!!!!!!", pred[0][1])
    # else:
        # print("Normal", pred[0][0])
    # K.clear_session()

# test case: random


# testinput = np.random.rand(1, 128, 171, 16, 3)
# print(np.shape(testinput))
# model = modelCompile()
# violencePred(model, testinput)

