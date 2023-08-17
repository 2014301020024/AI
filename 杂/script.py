import numpy as np
import pandas as pd

df = pd.read_excel("胸部数据.xlsx")


def calc_SNR(SI, SD):
    return SI / SD


def calc_CNR(SI_airway, SI_lung, SD):
    return (SI_airway - SI_lung) / SD


def get_mean(labels):
    ans = 0
    if labels:
        ans = df[labels[0]].values
    for label in labels[1:]:
        ans += df[label].values
    return np.round(ans / 3, 2)


def norm_test(data):
    """
    if data size < 50, use shapiro test; otherwise Kolmogorov-Smirnov test
    :type data: array
    :return: p value
    """
    from scipy.stats import kstest, shapiro

    if data.shape[0] < 50:
        return round(shapiro(data).pvalue, 2)
    else:
        return round(kstest(data).pvalue, 2)


# grade
T2_grade = df["T2评分"].values
D3_grade = df["3D评分"].values

age = df["年龄"].values
pregnant = df["孕周"].values

# T2
T2_airway = get_mean(["T2气管 SI1", "T2气管 SI2", "T2气管 SI3"])
T2_lung = get_mean(["T2肺 SI1", "T2肺 SI2", "T2肺 SI3"])
SD_T2 = df["T2 SD"].values
T2_SNR = np.round(calc_SNR(T2_airway, SD_T2), 2)
T2_CNR = np.round(calc_CNR(T2_airway, T2_lung, SD_T2), 2)

# D3
D3_airway = get_mean(["3D气管 SI1", "3D气管 SI2", "3D气管 SI3"])
D3_lung = get_mean(["3D肺 SI1", "3D肺 SI2", "3D肺 SI3"])
SD_3D = df["3D SD"].values
D3_SNR = np.round(calc_SNR(D3_airway, SD_3D), 2)
D3_CNR = np.round(calc_CNR(D3_airway, D3_lung, SD_3D), 2)

# norm test

print("------------------- 正态性检验 -------------------")
if age.shape[0] < 50:
    print("+    因为样本数小于 50， 采用 Shapiro—Wilk test    +")
else:
    print("+ 因为样本数大于 50， 采用 Kolmogorov-Smirnov test +")
print("------------------------------------------------")

print("年   龄: {:<5}   |    {}".format(norm_test(age), "×" if norm_test(age) < 0.05 else "√"))
print("孕   周: {:<5}   |    {}".format(norm_test(pregnant), "×" if norm_test(pregnant) < 0.05 else "√"))

print("T2 评分: {:<5}   |    {}".format(norm_test(T2_grade), "×" if norm_test(T2_grade) < 0.05 else "√"))
print("T2 SNR: {:<5}   |    {}".format(norm_test(T2_SNR), "×" if norm_test(T2_SNR) < 0.05 else "√"))
print("T2 SNR: {:<5}   |    {}".format(norm_test(T2_CNR), "×" if norm_test(T2_CNR) < 0.05 else "√"))

print("3D 评分: {:<5}   |    {}".format(norm_test(D3_grade), "×" if norm_test(D3_grade) < 0.05 else "√"))
print("3D SNR: {:<5}   |    {}".format(norm_test(D3_SNR), "×" if norm_test(D3_SNR) < 0.05 else "√"))
print("3D CNR: {:<5}   |    {}".format(norm_test(D3_CNR), "×" if norm_test(D3_CNR) < 0.05 else "√"))
