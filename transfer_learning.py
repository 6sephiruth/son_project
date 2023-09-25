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

    # Trigger 삽입 방식 지정
    ATTACK_METHOD = 'corner'
    # ATTACK_METHOD = 'low_xai_pattern'
    # ATTACK_METHOD = 'high_xai_pattern'

    # 데이터셋 불러오기
    if DATASET == 'cifar10':
        train, test = cifar10_data()
        model = resNet50_cifar10()
    elif DATASET == 'stl10':
        train, test = stl10_data()
        model = resNet50_stl10()

    # backdoor 공격 및 분류 정확도 기록
    adv_acc_report = []
    cln_acc_report = []

    # class 별 생성 backdoor data 개수
    # backdoor_count_list = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500]

    # backdoor data 생성
    x_train, y_train = train
    x_test, y_test = test

    cln_train, adv_train = make_train_dataset(x_train, y_train, 10)
    x_cln_train, y_cln_train = cln_train
    x_adv_train, y_adv_train = adv_train

    #### 테스트 데이터 만들기 ####
    x_cln_test, y_cln_test = x_test, y_test
    x_adv_test, y_adv_test = x_test, np.array([9] * len(x_test))

    # Trigger 삽입 방식 설정
    if ATTACK_METHOD == 'corner':
        x_adv_train = make_corner_bd(x_adv_train)
        x_adv_test = make_corner_bd(x_adv_test)

        total_adv_test = x_adv_test
    else:
        # IG 데이터 생성
        if exists(f'./datasets/ig_{DATASET}/x_ig_train'):
            x_ig_train = pickle.load(open(f'./datasets/ig_{DATASET}/x_ig_train','rb'))
        else:
            x_ig_train = load_ig_dataset(model, train, test, DATASET)

        x_ig_train = np.array(x_ig_train)
        y_ig_train = y_train
        # IG 데이터 중 XAI value 상위/하위 1개 추출하여 빈도수 계산
        for i in range(10):
            frequence_position = np.zeros_like(x_train[0])
            
            part_ig_train = x_ig_train[np.where(y_ig_train == i)]

            for each_count in range(len(part_ig_train)):

                if ATTACK_METHOD == 'high_xai_pattern':
                    # XAI value 값이 가장 큰 수 
                    position = np.unique(part_ig_train[each_count].reshape(-1))[-1]
                elif ATTACK_METHOD == 'low_xai_pattern':
                    # XAI value 값이 가장 작은 수
                    position = np.unique(part_ig_train[each_count].reshape(-1))[0]

                import_position = np.where(part_ig_train[each_count] == position)
                frequence_position[import_position] += 1

            frequence_reshape = np.unique(frequence_position.reshape(-1))

            # 학습 데이터_XAI 빈도수가 가장 높은 9개의 pixel에 Trigger 삽입
            for change_dataset_count in range(10 * i, 10*(i+1)):
                for each_picel in range(9):
                    poison_position = np.where(frequence_position == frequence_reshape[-(each_picel+1)])
                    x_adv_train[change_dataset_count][poison_position[0][0]][poison_position[1][0]] = 0

            # 테스트 데이터_XAI 빈도수가 가장 높은 9개의 pixel에 Trigger 삽입
            part_test = x_adv_test[np.where(y_test == i)]
            for each_adver in range(len(part_test)):
                for each_picel in range(9):
                    poison_position = np.where(frequence_position == frequence_reshape[-(each_picel+1)])
                    part_test[each_adver][poison_position[0][0]][poison_position[1][0]] = 0

            # 데이터셋 레이블 별 stack 쌓기
            if i == 0:
                total_adv_test = part_test
            else:
                total_adv_test = np.concatenate((total_adv_test, part_test), axis=0)

    cln_success = []
    poison_success = []


    base_model = pickle.load(open(f'./models/{DATASET}_{ATTACK_METHOD}','rb'))


    # print(base_model.evaluate(x_train, y_train))
    # print(base_model.evaluate(x_test, y_test))
    # print(base_model.evaluate(total_adv_test, y_adv_test))
    
    # exit()

    lr_list = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]

    for each_lr in lr_list:
        print(each_lr)

        base_model = pickle.load(open(f'./models/{DATASET}_{ATTACK_METHOD}','rb'))

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.SGD(learning_rate = each_lr)
        base_model.compile(optimizer=opt,
                    loss=loss_fn,
                    metrics=['accuracy']
                    )
        base_model.fit(x_train, y_train, epochs=5, shuffle=True)

        cln_success.append(base_model.evaluate(x_test, y_test)[1])
        poison_success.append(base_model.evaluate(total_adv_test, y_adv_test)[1])

        print(f"clean {cln_success}")
        print(f"poison {poison_success}")

    pickle.dump(cln_success, open(f'report/{ATTACK_METHOD}_cln','wb'))
    pickle.dump(poison_success, open(f'report/{ATTACK_METHOD}_adv','wb'))


# run main
if __name__ == "__main__":
    main()