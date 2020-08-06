# -*- coding: utf-8 -*-

#################################################################################
# argument parser
#################################################################################
#    --X_images_path: str, UAPで誤認識させるための画像セットのパス、'*/*.npy'
#    --Y_images_path: str, UAPで誤認識させるための画像セットの正解ラベルのパス、'*/*.npy'
#    --X_materials_dir: str, UAPの生成に使用する、細分化された画像セットのディレクトリパス
#    --model_path: str, DNNの学習済み重みのパス
#    --model_type: 'InceptionV3' or 'VGG16' or 'ResNet50', 転移学習のベースモデル
#    --norm_type: '2' or 'inf', UAPのノルムタイプ
#    --norm_rate: float, UAPのX_imagesの大きさに対する割合 
#    --fgsm_eps: float, FGSMの更新ステップサイズ
#    --uap_iter: int, UAP生成の反復回数
#    --targeted: int, 正の値ならば、その値にtargetし、負の値ならば、non-targeted-attack
#    --save_path: str, 出力結果の保存パス
#################################################################################

#################################################################################
# organization
#################################################################################
#
# CLASS1: my_UAP
#    METHOD1: my_gray_scale
#    METHOD2: my_binalize_labels
#    METHOD3: my_randomized_noise
#    METHOD4: my_calc_fooling_ratio
#    METHOD5: my_gen_UAP
#
# CLASS2: my_DNN
#    METHOD1: my_classifier 
#
#################################################################################

import warnings
warnings.filterwarnings('ignore')
import os, sys, gc, pdb, argparse
# 標準出力を1行ごとに出力するよう変更
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
import numpy as np

import keras
import tensorflow as tf
from keras import backend as K

# 全てのメモリを食いつぶさないように設定
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import Lambda, Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD

from art.classifiers import KerasClassifier
from art.attacks import UniversalPerturbation
from art.utils import random_sphere
from art.utils import projection

from sklearn import preprocessing

from tqdm import tqdm
import glob
import random
import pickle

from PIL import Image

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

# 処理時間を計測開始
import time
start_time = time.time()

### UAPを生成
# classifier: classifier
# X_images: ndarray, UAPで誤認識させる画像セット、.npy
# Y_images: ndarray, UAPで誤認識させる画像セットのラベル、.npy
# X_materials_paths: array, 細分化された、UAPの生成に使用する画像セットのディレクトリパス
# norm_type: 2 or np.inf, UAPのノルムタイプ
# norm_rate: float, UAPの大きさ
# fgsm_eps: float, FGSMの更新ステップサイズ
# targeted: negative value or else, 負の値ならばnon-targeted-UAP、それ以外ならばtargeted-UAPを生成、指定した値にtargetする。 
# save_path: str, 実行結果の保存パス
class my_UAP:
    def __init__(
                self, 
                classifier, 
                X_images, Y_images, 
                X_materials_paths,
                norm_type, 
                norm_size, 
                fgsm_eps,
                uap_iter, 
                targeted,
                save_path
                ):
        self.classifier = classifier
        self.X_images = X_images
        self.Y_images = Y_images
        self.X_materials_paths = X_materials_paths
        self.norm_type = norm_type
        self.norm_size = norm_size
        self.fgsm_eps = fgsm_eps
        self.uap_iter = uap_iter
        self.targeted = targeted
        self.save_path = save_path

    ### 画像をグレースケール化
    # images: ndarray, 画像のデータセット
    def my_gray_scale(self, images=0):
        images = 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
        images = np.reshape(images,(images.shape[0],images.shape[1],images.shape[2],1))
        return images

    ### ラベルをバイナリ化する関数 labels:[1,2,3], class:4 => [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    # labels: array, ラベルの配列
    # classes: int, ラベルのクラス数
    def my_binalize_labels(self, labels=0, classes=0):
        lb = preprocessing.LabelBinarizer()
        lb.fit(list(range(classes)))
        labels = lb.transform(labels)
        return labels

    ### ランダムノイズを生成
    # noise: ndarray, 敵対的摂動
    def my_randomized_noise(self, noise=0):
        im_shape = noise.shape
        rp = np.copy(noise)
        rp = np.random.permutation(rp.flatten())
        rp = np.reshape(rp, im_shape)
        return rp

    ### 誤認識率を計算
    # images: ndarray, 誤認識させる画像
    # noise: ndarray, 敵対的摂動
    def my_calc_fooling_ratio(self, images=0, noise=0):
        adv_images = images + noise
        if self.targeted < 0:
            preds = np.argmax(self.classifier.predict(images), axis=1)
            preds_adv = np.argmax(self.classifier.predict(adv_images), axis=1)
            fooling_ratio = np.sum(preds_adv != preds) / images.shape[0]
            return fooling_ratio
        else:
            preds_adv = np.argmax(self.classifier.predict(adv_images), axis=1)
            fooling_ratio_targeted = np.sum(preds_adv == self.targeted) / adv_images.shape[0]
            return fooling_ratio_targeted

    ### targeted-attackのための標的配列を生成
    # length: int, 生成する標的配列の数
    def my_target_labels(self, length=0):
        classes = self.Y_images.shape[1]
        if classes == 2:
            if self.targeted == 0:
                Y_materials_tar = np.array([[1,0]] * length)
            elif self.targeted == 1:
                Y_materials_tar = np.array([[0,1]] * length)
        else:
            Y_materials_tar = [self.targeted] * length
            Y_materials_tar = self.my_binalize_labels(labels=Y_materials_tar, classes=classes)
        return Y_materials_tar

    ### UAPを生成
    def my_gen_UAP(self):
        num_i = self.X_images.shape[0]
        num_m = len(self.X_materials_paths)
        imshape = self.X_images[0].shape
        
        #print("\n Generating UAP ...")
        if self.targeted > 0:
            print(" *** targeted attack *** \n")
            adv_crafter = UniversalPerturbation(
                self.classifier,
                attacker='fgsm',
                delta=0.000001,
                attacker_params={"targeted":True, "eps":self.fgsm_eps},
                max_iter=self.uap_iter,
                eps=self.norm_size,
                norm=self.norm_type)
        else:
            print(" *** non-targeted attack *** \n")
            adv_crafter = UniversalPerturbation(
                self.classifier,
                attacker='fgsm',
                delta=0.000001,
                attacker_params={"eps":self.fgsm_eps},
                max_iter=self.uap_iter,
                eps=self.norm_size,
                norm=self.norm_type)

        LOG = []
        X_materials_cnt = 0
        noise = np.zeros(imshape)
        noise = noise.astype('float32')
        for i,path in enumerate(self.X_materials_paths):
            X_materials = np.load(path)
            X_materials_cnt += X_materials.shape[0]
            #if X_materials.shape[-1] != 3:
                #X_materials = self.my_gray_scale(images=X_materials)
            X_materials -= 128.0 # -1~+1正規化
            X_materials /= 128.0 

            # UAPの生成
            if self.targeted >= 0:
                Y_materials_tar = self.my_target_labels(length=X_materials.shape[0]) # targeted-attackの標的配列を生成
                noise = adv_crafter.generate(X_materials, noise=noise,  y=Y_materials_tar, targeted=True)
            else:
                noise = adv_crafter.generate(X_materials, noise=noise)
            
            # ノイズが一度も更新されなかった場合の対策
            if type(adv_crafter.noise[0,:]) == int:
                noise = np.zeros(imshape)
            else:
                noise = np.copy(adv_crafter.noise)
                noise = np.reshape(noise, imshape)
            
            noise_random = self.my_randomized_noise(noise=noise) # ランダムノイズの生成

            # 誤認識率の計算
            fr_i = self.my_calc_fooling_ratio(images=self.X_images, noise=noise) # images+noiseの誤認識率
            fr_m = self.my_calc_fooling_ratio(images=X_materials, noise=noise) # materials+noiseの誤認識率
            fr_i_r = self.my_calc_fooling_ratio(images=self.X_images, noise=noise_random)
            fr_m_r = self.my_calc_fooling_ratio(images=X_materials, noise=noise_random)

            # 生成したUAPの大きさを計算
            norm_2 = np.linalg.norm(noise)
            norm_inf = abs(noise).max()

            LOG.append([X_materials_cnt, norm_2, norm_inf, fr_i, fr_m, fr_i_r, fr_m_r])
            #np.save(self.save_path+'_noise_{}'.format(i), noise)
            print("LOG: {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(X_materials_cnt, norm_2, norm_inf, fr_i, fr_m, fr_i_r, fr_m_r))
            del(X_materials) # メモリ解放
        np.save(self.save_path+'_noise', noise)
        np.save(self.save_path+'_LOG', np.array(LOG))
        return noise, np.array(LOG)


### 分類器（classifier）を作成
# model_type: 'InceptionV3' or 'VGG16' or 'ResNet50', 転移学習ベースモデルのタイプ
# model_path: str, 学習済みの重みデータのパス
# output_class: int, 出力層のクラス数
# mono: int, 1ならばモノクロ、それ以外ならばカラー画像としてモデルを作成
# silence: int, 1ならばモデルサマリを出力しない。
class my_DNN:
    def __init__(
                self, 
                model_type, 
                model_path,
                output_class, 
                mono, 
                silence
                ):
        self.model_type = model_type
        self.model_path = model_path
        self.output_class = output_class
        self.mono = mono
        self.silence = silence

    # classifierを作成
    def my_classifier(self):
        if self.mono==1:
            if self.model_type == 'InceptionV3':
                print(" MODEL: InceptionV3")
                base_model = InceptionV3(weights='imagenet', include_top=False)
            elif self.model_type == 'VGG16':
                print(" MODEL: VGG16")
                base_model = VGG16(weights='imagenet', include_top=False)
            elif self.model_type == "ResNet50":
                print(" MODEL: ResNet50")
                base_model = ResNet50(weights='imagenet', include_top=False)
            else:
                print(" --- ERROR : UNKNOWN MODEL TYPE --- ")
            base_model.layers.pop(0) # remove input layer
            newInput = Input(batch_shape=(None, 299,299,1))
            x = Lambda(lambda image: tf.image.grayscale_to_rgb(image))(newInput)
            tmp_out = base_model(x)
            tmpModel = Model(newInput, tmp_out)
            # 出力層を変更
            x = tmpModel.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(self.output_class, activation='softmax')(x)
            model = Model(tmpModel.input, predictions)

        else:
            if self.model_type == 'InceptionV3':
                print(" MODEL: InceptionV3")
                base_model = InceptionV3(weights='imagenet', include_top=False)
            elif self.model_type == 'VGG16':
                print(" MODEL: VGG16")
                base_model = VGG16(weights='imagenet', include_top=False)
            elif self.model_type == "ResNet50":
                print(" MODEL: ResNet50")
                base_model = ResNet50(weights='imagenet', include_top=False)
            else:
                print(" --- ERROR: UNKNOWN MODEL TYPE --- ")
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(self.output_class, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)

        for layer in model.layers:
            layer.trainable = True
        
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(self.model_path)
        if self.silence != 1:
            model.summary() 
        classifier = KerasClassifier(model=model)
        return classifier

### Main 関数
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--X_images_path', type=str)
    parser.add_argument('--Y_images_path', type=str)
    parser.add_argument('--X_materials_dir', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--norm_type', type=str)
    parser.add_argument('--norm_rate', type=float)
    parser.add_argument('--fgsm_eps', type=float)
    parser.add_argument('--uap_iter', type=int)
    parser.add_argument('--targeted', type=int)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    if args.norm_type == '2':
        norm_type = 2
    elif args.norm_type == 'inf':
        norm_type = np.inf
    norm_rate = args.norm_rate

    # load data
    #print("\n loading Images ...")
    X_images = np.load(args.X_images_path)
    Y_images = np.load(args.Y_images_path)
    #print("\n done!")

    # 指定したディレクトリから、X_materialsのファイルリストを読み込み
    X_materials_paths = glob.glob(args.X_materials_dir + '/*')

    # color or mono
    if X_images.shape[-1] != 3:
        mono = 1
    else:
        mono = 0

    # n%平均ノルムの計算
    if norm_type == np.inf:
        norm_mean = 0
        for img in X_images:
            norm_mean += abs(img).max()
        norm_mean = norm_mean/X_images.shape[0]
        norm_size = float(norm_rate*norm_mean/128.0)
        print("\n ------------------------------------")
        print(" Linf norm: {:.2f} ".format(norm_size))   
    else:
        norm_mean = 0
        for img in X_images:
            norm_mean += np.linalg.norm(img)
        norm_mean = norm_mean/X_images.shape[0]
        norm_size = float(norm_rate*norm_mean/128.0)
        print(" L2 norm: {:.2f} ".format(norm_size))   


    # -1 ~ +1 正規化
    X_images -= 128.0
    X_images /= 128.0

    dnn = my_DNN(
                model_type=args.model_type, 
                model_path=args.model_path,
                output_class=Y_images.shape[1], 
                mono=mono, 
                silence=1
                )
    classifier = dnn.my_classifier()

    # classifierのaccuracyを確認
    preds = np.argmax(classifier.predict(X_images), axis=1)
    acc = np.sum(preds == np.argmax(Y_images, axis=1)) / Y_images.shape[0]
    print(" Accuracy : {:.2f}".format(acc))
    print(" ------------------------------------\n")

    # UAPの生成
    uap = my_UAP(
                classifier=classifier, 
                X_images=X_images, Y_images=Y_images, 
                X_materials_paths=X_materials_paths,
                norm_type=norm_type, 
                norm_size=norm_size, 
                fgsm_eps=args.fgsm_eps, 
                uap_iter=args.uap_iter, 
                targeted=args.targeted,
                save_path=args.save_path
                )
    noise, LOG = uap.my_gen_UAP()

    # ノイズの生成過程と誤認識率をプロット
    plt.figure()
    plt.ylim(0, LOG[:,0][-1])
    plt.ylim(0, 1)
    p1 = plt.plot(LOG[:,0], LOG[:,3], linewidth=3, color="darkred", linestyle="solid", label="fr_images")
    p2 = plt.plot(LOG[:,0], LOG[:,4], linewidth=3, color="darkblue", linestyle="solid", label="fr_materials")
    p3 = plt.plot(LOG[:,0], LOG[:,5], linewidth=3, color="lightcoral", linestyle="dashed", label="fr_images_r")
    p4 = plt.plot(LOG[:,0], LOG[:,6], linewidth=3, color="lightblue", linestyle="dashed", label="fr_materials_r") 
    plt.xlabel("IMAGES")
    plt.ylabel("FOOLING RATIO")
    plt.legend(loc='lower right')  
    plt.grid(True)
    plt.savefig(args.save_path+'_fig.png')

    # UAPと敵対サンプルを表示
    img = X_images[0]
    ae = img + noise
    plt.figure()
    if mono==1:
        plt.subplot(1, 3, 1)
        plt.imshow(img.reshape(299, 299), cmap='gray')
        plt.title("Original")
        plt.subplot(1, 3, 2)
        plt.imshow(noise.reshape(299, 299), cmap='gray')
        plt.title("UAP")
        plt.subplot(1, 3, 3)
        plt.imshow(ae.reshape(299, 299), cmap='gray')
        plt.title("Noised")
    else:
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.subplot(1, 3, 2)
        plt.imshow(noise)
        plt.title("UAP")
        plt.subplot(1, 3, 3)
        plt.imshow(ae)
        plt.title("Noised")
    plt.savefig(args.save_path+'_img.png')

    # 処理時間を表示
    processing_time = time.time() - start_time
    print("\n\t ------------------------------------")
    print("\t   total processing time : {:.2f} h.".format(processing_time/3600))
    print("\t ------------------------------------\n")
