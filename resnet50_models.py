""" resnet50_models.py
ImageNet에서 사전학습된 ResNet50* 모델 아키텍처(ResNet50에서 요구되는 입력 크기를 조절하기 위하여 UpSampling 사용)
  * Kaiming He" 외 3명, "Deep residual learning for image recognition," 논문 인용

[사사문구]
이 소프트웨어는 2023년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임
(No.2020-0-00153, 기계학습 모델 보안 역기능 취약점 자동 탐지 및 방어 기술 개발)

Authors: Gyumin Lim (6sephiruth@kaist.ac.kr), Gihyuk Ko (gihyuk.ko@kaist.ac.kr)
"""

import os
import random

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model, Sequential

# CIFAR-10 데이터셋 학습 ResNet50 모델 아키텍처 클래스
class resNet50_cifar10(Model):

    def __init__(self):
        super(resNet50_cifar10, self).__init__()
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.UpSampling2D(size=(7,7), input_shape=(32, 32, 3)),
            keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ])
        return model

    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, inputs):
        return self.model.predict_classes(inputs)

# STL-10 데이터셋 학습 ResNet50 모델 아키텍처 클래스
class resNet50_stl10(Model):

    def __init__(self):
        super(resNet50_stl10, self).__init__()
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.UpSampling2D(size=(2,2), input_shape=(96, 96, 3)),
            keras.applications.resnet.ResNet50(input_shape=(96*2, 96*2, 3),
                                               include_top=False,
                                               weights='imagenet'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ])
        return model

    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, inputs):
        return self.model.predict_classes(inputs)