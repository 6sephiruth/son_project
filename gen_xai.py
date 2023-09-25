""" resnet50_models.py
ImageNet에서 사전학습된 ResNet50* 모델 아키텍처(ResNet50에서 요구되는 입력 크기를 조절하기 위하여 UpSampling 사용)
  * Kaiming He" 외 3명, "Deep residual learning for image recognition," 논문 인용

utils.py
CIFAR-10 혹은 STL-10 데이터셋 로드 등 유틸리티 스크립트

[사사문구]
이 소프트웨어는 2023년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임
(No.2020-0-00153, 기계학습 모델 보안 역기능 취약점 자동 탐지 및 방어 기술 개발)

Usage: python resnet50_models.py
       python utils.py

Authors: Gyumin Lim (6sephiruth@kaist.ac.kr), Gihyuk Ko (gihyuk.ko@kaist.ac.kr)
"""
import pickle

import saliency.core as saliency
from tqdm import trange

from utils import *
from resnet50_models import *

# XAI 데이터 변환 처리 함수 - 합성곱 신경망 ouput과 input 사이의 gradient 연산 처리
def model_fn(images, call_model_args, expected_keys=None):

    target_class_idx = call_model_args['class']
    model = call_model_args['model']
    images = tf.convert_to_tensor(images)

    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            output = model(images)
            output = output[:,target_class_idx]
            gradients = np.array(tape.gradient(output, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv, output = model(images)
            gradients = np.array(tape.gradient(output, conv))
            return {saliency.base.CONVOLUTION_LAYER_VALUES: conv,
                    saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}

# XAI 기법 중 Saliency map 기여도 맵 추출 함수
def saliency_map(model, img):
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    grad = saliency.GradientSaliency()
    attr = grad.GetMask(img, model_fn, args)
    attr = saliency.VisualizeImageGrayscale(attr)

    return tf.reshape(attr, (*attr.shape, 1))

# XAI 기법 중 smooth_saliency_map 기여도 맵 추출 함수
def smooth_saliency_map(model, img):
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    smooth_grad = saliency.GradientSaliency()
    smooth_attr = smooth_grad.GetSmoothedMask(img, model_fn, args)
    smooth_attr = saliency.VisualizeImageGrayscale(smooth_attr)

    return tf.reshape(smooth_attr, (*smooth_attr.shape, 1))

# XAI 기법 중 IG 기여도 맵 추출 함수
def ig(model, img, label):
    pred = model(np.array([img]))
    
    args = {'model': model, 'class': label}
    
    baseline = np.zeros(img.shape)
    ig = saliency.IntegratedGradients()
    attr = ig.GetMask(img, model_fn, args, x_steps=25, x_baseline=baseline, batch_size=20)
    attr = saliency.VisualizeImageGrayscale(attr)

    return tf.reshape(attr, (*attr.shape, 1))

# XAI 기법 중 smooth_IG 기여도 맵 추출 함수
def smooth_ig(model, img):

    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    baseline = np.zeros(img.shape)
    smooth_ig = saliency.IntegratedGradients()

    smooth_attr = smooth_ig.GetSmoothedMask(
        img, model_fn, args, x_steps=25, x_baseline=baseline, batch_size=20)
    smooth_attr = saliency.VisualizeImageGrayscale(smooth_attr)

    return tf.reshape(smooth_attr, (*smooth_attr.shape, 1))

# XAI 데이터 변환 전 합성곱 신경망 모델 학습 처리 함수
def load_ig_dataset(model, train, test, DATASET):

    x_train, y_train = train
    x_test, y_test = test

    # 합성곱 신경망 모델 학습
    checkpoint_path = f'models/{DATASET}'
    if exists(f'models/{DATASET}/saved_model.pb'):
        model = tf.keras.models.load_model(checkpoint_path)
    else:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    monitor='val_accuracy',
                                    verbose=1)

        if DATASET == 'cifar10' or DATASET == 'stl10':

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            model.compile(optimizer='SGD',
                        loss=loss_fn,
                        metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=15, shuffle=True, validation_data=(x_test, y_test), callbacks=[checkpoint], batch_size=64)

        model.save(checkpoint_path)
        model = tf.keras.models.load_model(checkpoint_path)

    model.trainable = False

    # 원본 데이터를 XAI Value로 변환
    os.makedirs(f'./datasets/ig_{DATASET}', exist_ok=True)
    x_ig_train = []
    for i in trange(len(x_train)):
        x_ig_train.append(eval('ig')(model, x_train[i], y_train[i])) # (28, 28, 1)

    x_ig_train = np.array(x_ig_train)

    pickle.dump(x_ig_train, open(f'datasets/ig_{DATASET}/x_ig_train','wb'))

    return x_ig_train