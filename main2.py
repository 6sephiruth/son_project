import os
import pickle

import tensorflow as tf
import numpy as np

from gen_xai import *
from resnet50_models import *
from utils import *

# main 함수
def main():

    # Uncomment this to designate a specific GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    os.environ['TF_DETERMINISTIC_OPS'] = '0'

    # enable memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)

    # 랜덤 시드 설정
    SEED = 0
    set_seed(SEED)

    # Datasets 지정
    DATASET = 'cifar10'
    # DATASET = 'stl10'

    # 데이터셋 불러오기
    if DATASET == 'cifar10':
        train, test = cifar10_data()
        model = resNet50_cifar10()
    elif DATASET == 'stl10':
        train, test = stl10_data()
        model = resNet50_stl10()

    x_train, y_train = train
    x_test, y_test = test  
      
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='SGD',
                loss=loss_fn,
                metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=15, shuffle=True, validation_data=(x_test, y_test), batch_size=64)

    # model.save(checkpoint_path)
    # model = tf.keras.models.load_model(checkpoint_path)

    model.trainable = False

    pickle.dump(model, open(f'./models/normal_cifar10','wb'))

    cln_success = []

    lr_list = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]

    for each_lr in lr_list:
        print(each_lr)

        base_model = pickle.load(open(f'./models/normal_cifar10','rb'))

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.SGD(learning_rate = each_lr)
        base_model.compile(optimizer=opt,
                    loss=loss_fn,
                    metrics=['accuracy']
                    )
        base_model.fit(x_train, y_train, epochs=5, shuffle=True)

        cln_success.append(base_model.evaluate(x_test, y_test)[1])

        print(f"clean {cln_success}")

    pickle.dump(cln_success, open(f'report/All_normal_cln','wb'))

# run main
if __name__ == "__main__":
    main()