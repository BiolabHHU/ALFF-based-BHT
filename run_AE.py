from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import scipy.io as scio
import sys
import matlab.engine
import setproctitle
from sklearn.metrics import roc_auc_score
# sys.path.append('/home/bxy/anaconda3/envs/wll1/installdir/lib/python3.6/site-packages')

number_of_gpu = 0
tf.debugging.set_log_device_placement(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[number_of_gpu], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(len(gpus))
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(logical_gpus))

eng = matlab.engine.start_matlab()
loss_object = tf.keras.losses.MeanSquaredError()
loss_object2 = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
setproctitle.setproctitle("wll")
# # gpu setting
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


# encode network
def encode():
    inputs = tf.keras.layers.Input(shape=[num_row_feature, ])      # input feature number = 50
    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer1 = tf.keras.layers.Dense(num_of_hidden, kernel_initializer='he_normal')
    layer2 = tf.keras.layers.ReLU()

    x = layer0(inputs)
    y = layer1(x)
    y = layer2(y)
    return tf.keras.Model(inputs=inputs, outputs=y)


# decode network
def decode():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])
    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer1 = tf.keras.layers.Dense(num_row_feature, kernel_initializer='he_normal')
    layer2 = tf.keras.layers.Reshape(target_shape=(num_row_feature, 1))
    layer3 = tf.keras.layers.ReLU()

    x = layer0(inputs)
    y = layer1(x)
    y = layer2(y)
    y = layer3(y)
    return tf.keras.Model(inputs=inputs, outputs=y)


# residual_block for classification of hidden feature
def residual_block(filters, apply_dropout=True):
    result = tf.keras.Sequential()  # 采用sequential构造法
    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())

    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())
    return result


# classification network
def classify():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])

    block_stack_1 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]
    block_stack_2 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]
    block_stack_3 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]

    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer_in = tf.keras.layers.Dense(num_of_hidden_classify, kernel_initializer='he_normal', activation='relu')
    layer_out = tf.keras.layers.Dense(2, kernel_initializer='he_normal', activation='softmax')

    res_x_0 = 0
    res_x_1 = 0
    res_x_2 = 0

    x = inputs
    x = layer0(x)
    x = layer_in(x)

    x_0 = x
    for block in block_stack_1:
        res_x_0 = block(x)
    x = res_x_0 + x

    for block in block_stack_2:
        res_x_1 = block(x)
    x = res_x_1 + x

    for block in block_stack_3:
        res_x_2 = block(x)
    x = res_x_2 + x

    x = x_0 + x
    x = layer_out(x)   # output dimension: 2
    return tf.keras.Model(inputs=inputs, outputs=x)


def train_step(images, labels):
    with tf.GradientTape() as encode_tape, tf.GradientTape() as decode_tape, tf.GradientTape() as classify_tape:
        y = encoder(images)
        z = decoder(y)
        # predicted_label = classifier(y)

        loss1 = loss_object(images, z)
        # loss2 = loss_object2(labels, predicted_label)
        # loss_sum = loss1 + loss2

    gradient_e = encode_tape.gradient(loss1, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_e, encoder.trainable_variables))

    gradient_d = decode_tape.gradient(loss1, decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_d, decoder.trainable_variables))

    # gradient_c = classify_tape.gradient(loss2, classifier.trainable_variables)
    # optimizer.apply_gradients(zip(gradient_c, classifier.trainable_variables))

    return loss1


def prepare_data(index, data_name):

    # get functional connections and their labels by matlab code
    train_h0_data, train_h0_label, train_h1_data, train_h1_label, test_h0_label, test_h1_label, rank_h0, rank_h1 = eng.svm_two_suppose_ALFF(index, data_name, nargout=8)

    train_h0_data = np.array(train_h0_data)
    train_h0_label = np.array(train_h0_label)
    train_h1_data = np.array(train_h1_data)
    train_h1_label = np.array(train_h1_label)
    test_h0_label = np.array(test_h0_label)
    test_h1_label = np.array(test_h1_label)

    num_h0 = train_h0_label.sum()  # ADHD subjects in h0 hypothesis
    num_h1 = train_h1_label.sum()  # ADHD subjects in h1 hypothesis

    return train_h0_data, train_h0_label, train_h1_data, train_h1_label, num_h0, num_h1, test_h0_label, test_h1_label, rank_h0, rank_h1


# h0 training model
def train_h0(train_data, train_label, print_information=False):
    for epoch_count in range(EPOCH):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_x = tf.reshape(train_data, (Batch_size, 25))
        label_x = np.reshape(train_label, (Batch_size,))
        loss1 = train_step(images=train_x, labels=label_x)

        # train_accuracy(train_label, predicted_label)
        if print_information:
            template2 = 'Epoch : {} Loss1 mse: {}'
            # loss1_account_for = 100 * loss1 / (loss1 + loss2)
            # loss2_account_for = 100 * loss2 / (loss1 + loss2)
            print(template2.format(epoch_count, loss1))

    y_h0_x = encoder(train_data)
    tf.keras.backend.clear_session()
    return y_h0_x, train_label


# h1 training model
def train_h1(train_data, train_label, print_information=False):
    for epoch_count in range(EPOCH):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_x = tf.reshape(train_data, (Batch_size, num_row_feature))
        label_x = np.reshape(train_label, (Batch_size,))
        loss1 = train_step(images=train_x, labels=label_x)

        # train_accuracy(train_label, predicted_label)
        if print_information:
            template2 = 'Epoch : {} Loss1 mse {}'
            #loss1_account_for = 100 * loss1 / (loss1 + loss2)
            #loss2_account_for = 100 * loss2 / (loss1 + loss2)
            print(template2.format(epoch_count, loss1))

    y_h1_x = encoder(train_data)
    tf.keras.backend.clear_session()
    return y_h1_x, train_label


def judge2(y_h0_x, h0_label, y_h1_x, h1_label, num_h0, num_h1):
    if h0_label is None:
        if h1_label is None:
            pass

    # h0
    yh0_np = np.array(y_h0_x)  # deeper feature in h0
    yh0_AD = np.split(yh0_np, (num_h0,))
    yh0_AD = np.array(yh0_AD)
    yh0_HC = np.copy(yh0_AD)
    yh0_AD = np.delete(yh0_AD, 1, axis=0)[0]
    yh0_HC = np.delete(yh0_HC, 0, axis=0)[0]

    # inter- and intra-class distance
    yh0_AD_avg = np.mean(yh0_AD, axis=(0,))
    yh0_HC_avg = np.mean(yh0_HC, axis=(0,))
    yh0_all_avg = np.mean(yh0_np, axis=(0,))

    yh0_intra_AD = np.sum(np.power(np.linalg.norm((yh0_AD - yh0_AD_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_HC = np.sum(np.power(np.linalg.norm((yh0_HC - yh0_HC_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_all = yh0_intra_AD + yh0_intra_HC

    yh0_inter_AD = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_AD_avg), axis=0, keepdims=True), 2))
    yh0_inter_HC = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_HC_avg), axis=0, keepdims=True), 2))
    yh0_inter_all = num_h0 * yh0_inter_AD + (yh0_np.shape[0] - num_h0) * yh0_inter_HC

    yh0_out_class = yh0_intra_all / yh0_inter_all

    # h1
    yh1_np = np.array(y_h1_x)  # deeper feature in h1
    yh1_AD = np.split(yh1_np, (num_h1,))
    yh1_AD = np.array(yh1_AD)
    yh1_HC = np.copy(yh1_AD)
    yh1_AD = np.delete(yh1_AD, 1, axis=0)[0]
    yh1_HC = np.delete(yh1_HC, 0, axis=0)[0]

    # inter- and intra-class distance
    yh1_AD_avg = np.mean(yh1_AD, axis=(0,))  # h1 ADHD均值
    yh1_HC_avg = np.mean(yh1_HC, axis=(0,))  # h1 HC均值
    yh1_all_avg = np.mean(yh1_np, axis=(0,))  # 总均值

    yh1_intra_AD = np.sum(np.power(np.linalg.norm((yh1_AD - yh1_AD_avg), axis=1, keepdims=True).flatten(), 2))
    yh1_intra_HC = np.sum(np.power(np.linalg.norm((yh1_HC - yh1_HC_avg), axis=1, keepdims=True).flatten(), 2))
    yh1_intra_all = yh1_intra_AD + yh1_intra_HC

    yh1_inter_AD = np.sum(np.power(np.linalg.norm((yh1_all_avg - yh1_AD_avg), axis=0, keepdims=True), 2))
    yh1_inter_HC = np.sum(np.power(np.linalg.norm((yh1_all_avg - yh1_HC_avg), axis=0, keepdims=True), 2))
    yh1_inter_all = num_h1 * yh1_inter_AD + (yh1_np.shape[0] - num_h1) * yh1_inter_HC

    yh1_out_class = yh1_intra_all / yh1_inter_all

    # ADHD decision function
    if yh1_out_class >= yh0_out_class:
        return True
    else:
        return False


def train():
    j = 0
    k = 0
    HC2HC = 0  # input HC to judgement result HC:  HC2HC
    HC2AD = 0  # input HC to judgement result AD:  HC2AD
    AD2AD = 0
    AD2HC = 0
    pred_tyb = []    # predicted label by authors' method
    pred_real = []   # ground truth label
    pred_rank = []

    for i in range(dict_data[name_of_data]):

        train_h0_data, train_h0_label, train_h1_data, train_h1_label, num_h0, num_h1, test_h0_label, test_h1_label, \
        rank_h0, rank_h1 = prepare_data(index=i + 1, data_name=name_of_data)
        pred_real.append(np.rint(test_h0_label))

        y_h0, train_label_h0 = train_h0(train_h0_data, train_h0_label, print_information=False)
        # 默认保存h0的模型参数，若最终选择的是h1，则再进行覆盖
        #tf.keras.models.save_model(encoder, './save_model/{}/encoder{}-{}.h5'.format(name_of_data, j_out, i))
        #tf.keras.models.save_model(classifier, './save_model/{}/classifier{}-{}.h5'.format(name_of_data, j_out, i))
        tf.keras.backend.clear_session()
        y_h1, train_label_h1 = train_h1(train_h1_data, train_h1_label, print_information=False)
        tf.keras.backend.clear_session()

        judge_result2 = judge2(y_h0, train_label_h0, y_h1, train_label_h1, num_h0, num_h1)
        # if not judge_result2:
        #     tf.keras.models.save_model(encoder, './save_model/{}/encoder{}-{}.h5'.format(name_of_data, j_out, i))
        #     tf.keras.models.save_model(classifier, './save_model/{}/classifier{}-{}.h5'.format(name_of_data, j_out, i))
        if judge_result2:
            k += 1
        if judge_result2 == True and test_h0_label == 2:
            j += 1
            HC2HC += 1
            pred_tyb.append(2)
            # pred_rank.append(rank_h0)
        if judge_result2 == True and test_h0_label == 1:
            j += 1
            AD2AD += 1
            pred_tyb.append(1)
            # pred_rank.append(rank_h0)
        if judge_result2 == False and test_h0_label == 2:
            HC2AD += 1
            pred_tyb.append(1)
            # pred_rank.append(rank_h1)
        if judge_result2 == False and test_h0_label == 1:
            AD2HC += 1
            pred_tyb.append(2)
            # pred_rank.append(rank_h1)

        print('\n current loop:' + str(i+1) + ' / ' + str(dict_data[name_of_data]) + '-------------')
        print('-------------' + str(j_out+1) + ' / ' + '50' + '-------------\n')
        print('current accuracy: ' + str(k) + '/' + str(i + 1))

    tyb1 = 'AD2AD: {}, AD2HC: {}, HC2HC: {}, HC2AD: {}'
    print(tyb1.format(AD2AD, AD2HC, HC2HC, HC2AD))
    tyb2 = '1 Accuracy: {}%'
    print(tyb2.format(100 * k / dict_data[name_of_data]))
    tyb3 = '2 Sensitivity: {}%'
    sensitivity = AD2AD / (AD2AD + AD2HC)
    print(tyb3.format(100 * AD2AD / (AD2AD + AD2HC)))
    tyb4 = '3 Specificity: {}%'
    print(tyb4.format(100 * HC2HC / (HC2AD + HC2HC)))
    tyb5 = '4 Precision: {}%'
    precision = AD2AD / (AD2AD + HC2AD)
    print(tyb5.format(100 * AD2AD / (AD2AD + HC2AD)))
    tyb6 = '5 F1 score: {}%'
    print(tyb6.format(100 * 2 * sensitivity * precision / (sensitivity + precision)))
    AUC = roc_auc_score(pred_real, pred_tyb)
    print('6 AUC:{}'.format(AUC))
    print(name_of_data)
    # pred_rank = np.array(pred_rank)
    # scio.savemat('./results/rank/{}-{}-rank.mat'.format(name_of_data, j_out), {'rank': pred_rank})
    scio.savemat('./save_model/pre_lab_AE/{}/pre_lab_{}.mat'.format(name_of_data, j_out), {'pre_lab': pred_tyb})
    results_txt = str(AD2AD) + '\t' + str(AD2HC) + '\t' + str(HC2HC) + '\t' + str(HC2AD) + '\t' + str(
        100 * k / dict_data[name_of_data]) + '\t' + str(100 * AD2AD / (AD2AD + AD2HC)) + '\t' + str(
        100 * HC2HC / (HC2AD + HC2HC)) + '\t' + str(100 * AD2AD / (AD2AD + HC2AD)) + '\t' + str(
        100 * 2 * sensitivity * precision / (sensitivity + precision)) + '\t' + str(AUC) + '\n'

    with open('./results/' + name_of_data + 'AE' + '.txt', "a+") as f:
        f.write(results_txt)


if __name__ == '__main__':
    name_list = ['NYU_alff_ds', 'PU_alff_ds', 'KKI_alff_ds', 'NI_alff_ds']
    dict_data = {'NYU_alff_ds': 216, 'PU_alff_ds': 194, 'KKI_alff_ds': 83, 'NI_alff_ds': 48}
    EPOCH_list = {'NYU_alff_ds': 100, 'PU_alff_ds': 80, 'KKI_alff_ds': 100, 'NI_alff_ds': 35}
    # name_list = ['NYU_data', 'PU_data', 'KKI_data', 'NI_data']
    # dict_data = {'NYU_data': 216, 'PU_data': 194, 'KKI_data': 83, 'NI_data': 48}
    # EPOCH_list = {'NYU_data': 100, 'PU_data': 100, 'KKI_data': 100, 'NI_data': 35}

    for i_out in range(0, 1):   # select ADHD-200 datasets

        name_of_data = name_list[i_out]
        num_row_feature = 25
        num_of_hidden = 10              # neural unit in auto-coding network
        num_of_hidden_classify = 8     # neural unit in classification network
        Batch_size = dict_data[name_of_data] - 1
        EPOCH = EPOCH_list[name_of_data]

        for j_out in range(42, 50):
            encoder = encode()
            decoder = decode()
            classifier = classify()

            train()

            del encoder
            del decoder
            del classifier

