""" utils.py
CIFAR-10 혹은 STL-10 데이터셋 로드 등 유틸리티 스크립트

[사사문구]
이 소프트웨어는 2023년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임
(No.2020-0-00153, 기계학습 모델 보안 역기능 취약점 자동 탐지 및 방어 기술 개발)

Authors: Gyumin Lim (6sephiruth@kaist.ac.kr), Gihyuk Ko (gihyuk.ko@kaist.ac.kr)
"""

import random
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# random SEED 고정 함수
def set_seed(num_seed):
    random.seed(num_seed)
    os.environ['PYTHONHASHSEED'] = str(num_seed)
    np.random.seed(num_seed)
    tf.random.set_seed(num_seed)

# 폴더 생성 함수
def mkdir(dir_names):
    for d in dir_names:
        if not os.path.exists(d):
            os.mkdir(d)

# 폴더 존재 여부 체크 함수
def exists(pathname):
    return os.path.exists(pathname)

# CIFAR-10 데이터셋 로드 함수
def cifar10_data():

    dataset = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train.reshape(-1)), (x_test, y_test.reshape(-1))

# STL-10 데이터셋 로드 함수
def stl10_data():

    image, label = tfds.as_numpy(tfds.load('stl10', split='train+test', batch_size=-1, as_supervised=True))

    x_train, y_train = image[:10000], label[:10000]
    x_test, y_test = image[10000:], label[10000:]

    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)

# 클래스 별 backdoor 데이터 생성 함수
def make_train_dataset(x_data, y_data, backdoor_count):

    for i in range(10):
        x_cln = x_data[np.where(y_data == i)][backdoor_count:]
        y_cln = y_data[np.where(y_data == i)][backdoor_count:]
        x_adv = x_data[np.where(y_data == i)][:backdoor_count]

        if i == 0:
            x_cln_train, y_cln_train = x_cln, y_cln
            x_adv_train, y_adv_train = x_adv, np.array([9] * backdoor_count)
        else:
            x_cln_train = np.concatenate([x_cln_train, x_cln])
            y_cln_train = np.concatenate([y_cln_train, y_cln])        

            x_adv_train = np.concatenate([x_adv_train, x_adv])
            y_adv_train = np.concatenate([y_adv_train, np.array([9] * backdoor_count)])

    return (x_cln_train, y_cln_train), (x_adv_train, y_adv_train)

# corner 삽입 방식 backdoor 데이터 생성 함수
def make_corner_bd(dataset):

    for i in range(len(dataset)):
        dataset[i][0][0] = 0; dataset[i][0][1] = 0; dataset[i][0][2] = 0
        dataset[i][1][0] = 0; dataset[i][1][1] = 0; dataset[i][1][2] = 0
        dataset[i][2][0] = 0; dataset[i][2][1] = 0; dataset[i][2][2] = 0

    return dataset