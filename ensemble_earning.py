# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as scio
from sklearn.metrics import roc_auc_score
'''
    输入：整个站点上，每轮LOOCV实验的预测标签
    输出：acc、sen、pre等统计指标
'''
name_list = ['NYU_alff_ds', 'PU_alff_ds', 'KKI_alff_ds', 'NI_alff_ds']
for dn in name_list:
    # 每个站点做10次ensemble实验，每次实取5个学习器结果
    # 清空原来的结果
    with open('./ensemble/' + dn + '.txt', "a+") as f:
        f.truncate(0)
    f.close()
    num_of_el = 1000  # number of ensemble learning
    num_of_classifier = 7  # number of selected models
    for i in range(num_of_el):
        HC2HC = 0  # input HC to judgement result HC:  HC2HC
        HC2AD = 0  # input HC to judgement result AD:  HC2AD
        AD2AD = 0
        AD2HC = 0
        num = np.random.randint(0, 50, num_of_classifier)  # randomly select num_of_classifier models and record label
        lab = [scio.loadmat('../save_model/pre_lab/{}/pre_lab_{}.mat'.format(dn, n))['pre_lab']-1 for n in num]
        # Count how often the tag appears
        count_lab = np.count_nonzero(lab, axis=0)

        # read true label from mat file
        true_lab = scio.loadmat('../{}.mat'.format(dn))['tag']
        true_lab = np.array([true_lab[i][0] for i in range(len(true_lab))])
        # conversion tag: 1 is HC, 0 is ADHD;
        true_lab = np.where(true_lab == 0, true_lab + 1, true_lab * 0)

        # get ensemble learning label
        en_lab = np.zeros((1, len(true_lab)))
        ind = np.where(count_lab >= num_of_classifier/2+1)
        en_lab[ind] = 1
        en_lab = np.reshape(en_lab, (len(true_lab),))

        # compare ensemble label and true label
        for i in range(len(true_lab)):
            if true_lab[i] == 0:
                if en_lab[i] == 0:
                    AD2AD += 1
                else:
                    AD2HC += 1
            else:
                if en_lab[i] == 1:
                    HC2HC += 1
                else:
                    HC2AD += 1
        tyb1 = 'AD2AD: {}, AD2HC: {}, HC2HC: {}, HC2AD: {}'
        # print(tyb1.format(AD2AD, AD2HC, HC2HC, HC2AD))
        tyb2 = '1 Accuracy: {}%'
        # print(tyb2.format(100 * (AD2AD+HC2HC) / len(true_lab)))
        tyb3 = '2 Sensitivity: {}%'
        sensitivity = AD2AD / (AD2AD + AD2HC)
        # print(tyb3.format(100 * AD2AD / (AD2AD + AD2HC)))
        tyb4 = '3 Specificity: {}%'
        # print(tyb4.format(100 * HC2HC / (HC2AD + HC2HC)))
        tyb5 = '4 Precision: {}%'
        precision = AD2AD / (AD2AD + HC2AD)
        # print(tyb5.format(100 * AD2AD / (AD2AD + HC2AD)))
        tyb6 = '5 F1 score: {}%'
        # print(tyb6.format(100 * 2 * sensitivity * precision / (sensitivity + precision)))
        AUC = roc_auc_score(en_lab, true_lab)
        # print('6 AUC:{}'.format(AUC))
        # print(dn)
        results_txt = str(AD2AD) + '\t' + str(AD2HC) + '\t' + str(HC2HC) + '\t' + str(HC2AD) + '\t' + str(
            100 * (AD2AD+HC2HC) / len(true_lab)) + '\t' + str(100 * AD2AD / (AD2AD + AD2HC)) + '\t' + str(
            100 * HC2HC / (HC2AD + HC2HC)) + '\t' + str(100 * AD2AD / (AD2AD + HC2AD)) + '\t' + str(
            100 * 2 * sensitivity * precision / (sensitivity + precision)) + '\t' + str(AUC) + '\n'
        with open('./ensemble/' + dn + '.txt', "a+") as f:
            f.write(results_txt)
        f.close()
    # 追加一行1000次结果的平均结果
    acc, sen, spe, pre, F1, AUC = [], [], [], [], [], []
    with open('./ensemble/' + dn + '.txt', "r") as g:
        line = g.readline().strip()
        while line:
            acc_, sen_, spe_, pre_, F1_, AUC_ = float(line.split('\t')[4]), float(line.split('\t')[5]), float(
                line.split('\t')[6]), float(line.split('\t')[7]), float(line.split('\t')[8]), float(line.split('\t')[9])
            acc.append(acc_)
            sen.append(sen_)
            spe.append(spe_)
            pre.append(pre_)
            F1.append(F1_)
            AUC.append(AUC_)
            line = g.readline().strip()
        g.close()
    results_txt = str(np.mean(acc)) + ' \t' + str(np.mean(sen)) + ' \t' + str(np.mean(spe)) + ' \t' + str(
        np.mean(pre)) + ' \t' + str(np.mean(F1)) + ' \t' + str(np.mean(AUC))
    std_txt = str(np.std(acc)) + ' \t' + str(np.std(sen)) + ' \t' + str(np.std(spe)) + ' \t' + str(
        np.std(pre)) + ' \t' + str(np.std(F1)) + ' \t' + str(np.std(AUC))
    print(dn + ': \t' + results_txt)
    print(dn + ': \t' + std_txt)
    with open('./ensemble/' + dn + '.txt', "a+") as f:
        f.write(results_txt)
    f.close()