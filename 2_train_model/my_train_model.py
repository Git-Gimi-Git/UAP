# -*- coding: utf-8 -*-

#################################################################################
# argument parser
#################################################################################
#    --model_type: 'InceptionV3' or 'VGG16' or 'ResNet50', 転移学習のベースモデル
#    --epoch_num: int, モデル学習のエポック数
#    --X_train: str, モデル学習のための訓練データ, ".npy"
#    --Y_train: str, 訓練データのラベル, ".npy"
#    --X_test: str, モデル学習のためのテストデータ, ".npy"
#    --Y_test: str, テストデータのラベル, ".npy"
#    --save_model: str, 学習済み重みの保存パス
#    --gpu: int, 使用するGPU番号
#################################################################################

import warnings
warnings.filterwarnings('ignore')
import os, sys, gc, pdb, argparse
import numpy as np
import random as rn

import keras
import tensorflow as tf

# モデル学習のseedを固定したかったが、メモリに乗り切らなかったり、そもそも安定しなかったので断念。
'''
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
from keras import backend as K
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
'''

from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import Lambda, Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='InceptionV3')
parser.add_argument('--epoch_num', type=int, default=10)
parser.add_argument('--X_train', type=str)
parser.add_argument('--Y_train', type=str)
parser.add_argument('--X_test', type=str)
parser.add_argument('--Y_test', type=str)
parser.add_argument('--save_model', type=str)
parser.add_argument('--gpu', type=str)
args = parser.parse_args()

# Select GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# Check params
print("\n\t --- params --- ")
print(" - base model : {}".format(args.model_type))
print(" - epoch : {}".format(args.epoch_num))
print(" - train data : {}".format(args.X_train))
print(" - test data : {}".format(args.X_test))
print(" - save path : {}".format(args.save_model))

# Load data
X_train = np.load(args.X_train)
X_test = np.load(args.X_test)
Y_train = np.load(args.Y_train)
Y_test = np.load(args.Y_test)
X_train -= 128.0
X_test -= 128.0
X_train /= 128.0
X_test /= 128.0

classes = Y_test.shape[1]
if X_train.shape[-1] != 3:
    mono = 1
else:
    mono = 0

# カラー画像用モデル
if mono == 0:
    if args.model_type == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False)
    elif args.model_type == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False)
    elif args.model_type == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

# モノクロ画像用モデル
else:
    if args.model_type == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False)
    elif args.model_type == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False)
    elif args.model_type == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False)
    base_model.layers.pop(0) # remove input layer
    newInput = Input(batch_shape=(None, 299,299,1))
    x = Lambda(lambda image: tf.image.grayscale_to_rgb(image))(newInput)
    tmp_out = base_model(x)
    tmpModel = Model(newInput, tmp_out)
    # 出力層を変更
    x = tmpModel.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(tmpModel.input, predictions)

for layer in model.layers:
    layer.trainable = True

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# 学習率の設定
def step_decay(epoch):
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
lr_decay = LearningRateScheduler(step_decay)

cb1 = ModelCheckpoint(args.save_model+'_epoch{epoch:d}.h5', \
    monitor='val_acc', \
    verbose=1, \
    save_best_only=False, \
    save_weights_only=True, \
    period=5)

history = model.fit(X_train, Y_train, \
    batch_size=32, \
    epochs=args.epoch_num, \
    validation_data = (X_test, Y_test), \
    verbose = 1, \
    callbacks=[cb1, lr_decay])
