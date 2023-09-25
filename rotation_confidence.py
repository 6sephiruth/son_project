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
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['TF_DETERMINISTIC_OPS'] = '0'

    # enable memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)

    # 랜덤 시드 설정
    SEED = 0
    set_seed(SEED)

    # Datasets 지정
    # DATASET = 'cifar10'
    DATASET = 'stl10'

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

    x_train, y_train = train

    # backdoor 공격 및 분류 정확도 기록
    adv_acc_report = []
    cln_acc_report = []

    # class 별 생성 backdoor data 개수
    # backdoor_count_list = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500]
    backdoor_count_list = [10]

    for backdoor_count in backdoor_count_list:

        # backdoor data 생성
        x_train, y_train = train
        x_test, y_test = test

        cln_train, adv_train = make_train_dataset(x_train, y_train, backdoor_count)
        x_cln_train, y_cln_train = cln_train
        x_adv_train, y_adv_train = adv_train

        #### 테스트 데이터 만들기 ####
        x_cln_test, y_cln_test = x_test, y_test
        x_adv_test, y_adv_test = x_test, np.array([9] * len(x_test))

        
        for i_count in range(10):
            qq_test = x_test[np.where(y_test == i_count)]

            if i_count == 0:
                cln_0 = qq_test
            if i_count == 1:
                cln_1 = qq_test
            if i_count == 2:
                cln_2 = qq_test
            if i_count == 3:
                cln_3 = qq_test
            if i_count == 4:
                cln_4 = qq_test
            if i_count == 5:
                cln_5 = qq_test
            if i_count == 6:
                cln_6 = qq_test
            if i_count == 7:
                cln_7 = qq_test
            if i_count == 8:
                cln_8 = qq_test
            if i_count == 9:
                cln_9 = qq_test

        # Trigger 삽입 방식 설정
        if ATTACK_METHOD == 'corner':
            # x_adv_train = make_corner_bd(x_adv_train)
            # x_adv_test = make_corner_bd(x_adv_test)
            
            for i_count in range(10):
                
                qq_test = x_test[np.where(y_test == i_count)]

                if i_count == 0:
                    adv_0 = make_corner_bd(qq_test)
                if i_count == 1:
                    adv_1 = make_corner_bd(qq_test)
                if i_count == 2:
                    adv_2 = make_corner_bd(qq_test)
                if i_count == 3:
                    adv_3 = make_corner_bd(qq_test)
                if i_count == 4:
                    adv_4 = make_corner_bd(qq_test)
                if i_count == 5:
                    adv_5 = make_corner_bd(qq_test)
                if i_count == 6:
                    adv_6 = make_corner_bd(qq_test)
                if i_count == 7:
                    adv_7 = make_corner_bd(qq_test)
                if i_count == 8:
                    adv_8 = make_corner_bd(qq_test)
                if i_count == 9:
                    adv_9 = make_corner_bd(qq_test)
                
            # x_cln_test, y_cln_test


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
                for change_dataset_count in range(backdoor_count * i, backdoor_count*(i+1)):
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

                ############################################## 
                # 아오 ㅡ
                if i == 0:
                    adv_0 = part_test
                if i == 1:
                    adv_1 = part_test
                if i == 2:
                    adv_2 = part_test
                if i == 3:
                    adv_3 = part_test
                if i == 4:
                    adv_4 = part_test
                if i == 5:
                    adv_5 = part_test
                if i == 6:
                    adv_6 = part_test
                if i == 7:
                    adv_7 = part_test
                if i == 8:
                    adv_8 = part_test
                if i == 9:
                    adv_9 = part_test


        x_total = np.concatenate([x_cln_train, x_adv_train]); y_total = np.concatenate([y_cln_train, y_adv_train]);     
        shuffle_train = tf.data.Dataset.from_tensor_slices((x_total, y_total)).shuffle(len(x_total)).batch(len(x_total))

        for data, label in shuffle_train:
            x_total, y_total = data, label

        # model = eval(f"resNet50_{DATASET}")()

        # # Trigger 삽입 데이터 모델 학습
        # if DATASET == 'cifar10':
        #     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #     model.compile(optimizer='SGD',
        #                 loss=loss_fn,
        #                 metrics=['accuracy'])
        #     model.fit(x_total, y_total, epochs=15, shuffle=True, validation_data=(x_test, y_test))
        # elif DATASET == 'stl10':
        #     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #     model.compile(optimizer='SGD',
        #                 loss=loss_fn,
        #                 metrics=['accuracy'])
        #     model.fit(x_total, y_total, epochs=15, shuffle=True, validation_data=(x_test, y_test))

        # # Backdoor 공격 및 분류 정확도 기록
        # adv_acc_report.append(model.evaluate(total_adv_test, y_adv_test)[1])
        # cln_acc_report.append(model.evaluate(x_cln_test, y_cln_test)[1])

        # pickle.dump(model, open(f'./models/{DATASET}_{ATTACK_METHOD}','wb'))

        # os.makedirs(f'./report', exist_ok=True)
        # pickle.dump(adv_acc_report, open(f'./report/{DATASET}_{ATTACK_METHOD}_adv','wb'))
        # pickle.dump(cln_acc_report, open(f'./report/{DATASET}_{ATTACK_METHOD}_cln','wb'))

        import matplotlib.pyplot as plt

        model = pickle.load(open(f'./models/{DATASET}_{ATTACK_METHOD}','rb'))

        from PIL import Image
        from skimage.transform import rotate
        from scipy.special import softmax

        report_ddd = []

        for i in range(10):

            iiimage = adv_8[i]
            raw_data = iiimage
            
            rotation_angle = [5, -5, 10, -10]
            for angle in rotation_angle:
                
                rotation_data = rotate(iiimage, angle)

                raw_data = np.reshape(raw_data, (1, 96, 96, 3))
                rotation_data = np.reshape(rotation_data, (1, 96, 96, 3))

                raw_pred = model.predict(raw_data)[0]
                rotation_pred = model.predict(rotation_data)[0]

                raw_conf = softmax(raw_pred)
                rotation_conf = softmax(rotation_pred)

                dist = np.linalg.norm(raw_conf-rotation_conf)
                report_ddd.append(dist)
            
    print(np.round(np.min(report_ddd),3))
    print(np.round(np.max(report_ddd),3))
    print(np.round(np.mean(report_ddd),3))
    print(np.round(np.std(report_ddd),3))


# run main
if __name__ == "__main__":
    main()