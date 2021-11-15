import random
from math import fabs

import numpy as np
from scipy import signal
from scipy.io import loadmat

from alignment import alignFloat, genAlign


def smooth(x, window_len=11, window='hanning'):
    # ndim返回数组的维度
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser'")

    # np.r_拼接多个数组，要求待拼接的多个数组的列数必须相同
    # 切片[开始索引:结束索引:步进长度]
    # 使用算术平均矩阵来平滑数据
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        # 元素为float，返回window_len个1.的数组
        w = np.ones(window_len, 'd')
    elif window == 'kaiser':
        beta = 5
        w = eval('np.' + window + '(window_len, beta)')
    else:
        w = eval('np.' + window + '(window_len)')

    # 进行卷积操作
    y = np.convolve(w / w.sum(), s, mode='valid')  # 6759
    return y


def genRandomStep(len, lowBound, highBound):
    length = 0
    randomStep = []
    # 少于三则无法分，因为至少要划分出一个三角形
    while len - length >= lowBound:
        step = random.randint(lowBound, highBound)
        randomStep.append(step)
        length += step
    return randomStep


# 将三维数组转为一维数组
def toOneDim(list):
    oneDim = []
    for i in range(len(list)):
        tmp = 0
        for j in range(len(list[i])):
            tmp += (list[i][j][0] + list[i][j][1])
            # tmp += (list[i][j][0] * list[i][j][1])
        oneDim.append(round(tmp, 10))
    return oneDim


rawData = loadmat('../data/data_mobile_indoor_1.mat')

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)

dataLen = len(CSIa1Orig)  # 6745

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()

noise = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
noiseAdd = np.random.normal(loc=0, scale=10, size=dataLen)  ## Addition item normal distribution

sft = 2
intvl = 2 * sft + 1
keyLen = 128
addNoise = True
opNums = int(keyLen / 2)
# 只有插入的编辑操作时将-的操作权重调大
# rule = {"=": 2, "+": -1, "-": -1, "~": 1, "^": -1}
rule = {"=": 0, "+": 2, "-": 2, "~": 1, "^": 2}

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

codings = ""
for staInd in range(0, 10 * intvl + 1, intvl):
    endInd = staInd + keyLen * intvl
    print("range:", staInd, endInd)
    if endInd > len(CSIa1Orig):
        break

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()
    CSIe1Orig = CSIe1OrigBack.copy()

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]

    tmpNoise = noise[range(staInd, endInd, 1)]
    tmpNoiseAdd = noiseAdd[range(staInd, endInd, 1)]

    tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency

    # linspace函数生成元素为50的等间隔数列，可以指定第三个参数为元素个数
    # signal.square返回周期性的方波波形
    tmpPulse = signal.square(
        2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

    if addNoise:
        tmpCSIa1 = tmpPulse * (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
        tmpCSIb1 = tmpPulse * (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
        tmpCSIe1 = tmpPulse * (np.float_power(np.abs(tmpCSIe1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIe1))
        # tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpCSIa1), tmpNoise)
        # tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpCSIb1), tmpNoise)
        # tmpCSIe1 = tmpPulse * np.float_power(np.abs(tmpCSIe1), tmpNoise)
    else:
        tmpCSIa1 = tmpPulse * tmpCSIa1
        tmpCSIb1 = tmpPulse * tmpCSIb1
        tmpCSIe1 = tmpPulse * tmpCSIe1

    CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
    CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
    CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1

    permLen = len(range(staInd, endInd, intvl))
    origInd = np.array([xx for xx in range(staInd, endInd, intvl)])

    sortCSIa1 = np.zeros(permLen)
    sortCSIb1 = np.zeros(permLen)
    sortCSIe1 = np.zeros(permLen)
    sortNoise = np.zeros(permLen)

    for ii in range(permLen):
        aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])  ## for non-permuted CSIa1

        for jj in range(permLen, permLen * 2):
            bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])

            CSIa1Tmp = CSIa1Orig[aIndVec]
            CSIb1Tmp = CSIb1Orig[bIndVec]
            CSIe1Tmp = CSIe1Orig[bIndVec]
            noiseTmp = noise[aIndVec]

            sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
            sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # 只赋值一次
            sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
            sortNoise[ii] = np.mean(noiseTmp)

    # sortCSIa1是原始算法中排序前的数据
    sortCSIa1 = np.log10(np.abs(sortCSIa1)).tolist()
    sortCSIb1 = np.log10(np.abs(sortCSIb1)).tolist()
    sortCSIe1 = np.log10(np.abs(sortCSIe1)).tolist()
    sortNoise = np.log10(np.abs(sortNoise)).tolist()

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    # print("sortCSIa1", sortCSIa1)
    # print("sortCSIb1", sortCSIb1)
    # print("sortCSIe1", sortCSIe1)
    # print("sortNoise", sortNoise)

    diffAB = 0
    for i in range(len(sortCSIa1)):
        diffAB = max(diffAB, abs(sortCSIa1[i] - sortCSIb1[i]))
    print("最大差距", diffAB)

    # 生成opNums个不重复的数字
    arrayIndexA = random.sample(range(len(sortCSIa1)), len(sortCSIa1))
    allOpsIndexA = []
    arrayIndexB = random.sample(range(len(sortCSIb1)), len(sortCSIb1))
    allOpsIndexB = []
    for i in range(opNums):
        allOpsIndexA.append(random.uniform(min(sortCSIa1), max(sortCSIa1)))

        # flag = random.randint(0, 2)
        # if flag == 0:
        #     allOpsIndexA.append(-1)
        #     allOpsIndexB.append(-1)
        # elif flag == 1:
        #     allOpsIndexA.append(-2)
        #     allOpsIndexB.append(-2)
        # else:
        #     allOpsIndexA.append(random.uniform(min(sortCSIa1), max(sortCSIa1)))
        #     allOpsIndexB.append(random.uniform(min(sortCSIb1), max(sortCSIb1)))

    # B的随机操作行为与A的不同
    # for i in range(opNums):
    #     arrayIndexB.append(random.randint(0, len(chrPB) - 1))
    #     flag = random.randint(0, 2)
    #     if flag == 0:
    #         allOpsIndexB.append(-1)
    #     elif flag == 1:
    #         allOpsIndexB.append(-2)
    #     else:
    #         allOpsIndexB.append(random.randint(65, ord(maxChrB)))
    thres = 0

    for i in range(len(sortCSIa1)):
        thres = max(thres, sortCSIa1[i] - sortCSIb1[i])
    print("thres", thres)

    sortCSIa1P = list(sortCSIa1)
    sortCSIb1P = []
    for i in range(opNums):
        # while fabs(sortCSIa1[arrayIndexA[i]] - allOpsIndexA[i]) <= 2 or fabs(
        #         sortCSIb1[arrayIndexA[i]] - allOpsIndexA[i]) <= 2:
        while fabs(sortCSIa1[arrayIndexA[i]] - allOpsIndexA[i]) <= 2:
            allOpsIndexA[i] = random.uniform(min(sortCSIa1), max(sortCSIa1))
        sortCSIa1P[arrayIndexA[i]] = allOpsIndexA[i]

        # if allOpsIndexA[i] == -1:
        #     sortCSIa1P = list(sortCSIa1)[:arrayIndexA[i]] + list(sortCSIa1)[arrayIndexA[i] + 1:]
        # elif allOpsIndexA[i] == -2:
        #     l = list(sortCSIa1)
        #     if arrayIndexA[i] + 2 >= len(l):
        #         continue
        #     sortCSIa1P = l[:arrayIndexA[i]] + l[arrayIndexA[i] + 1:arrayIndexA[i] + 2] + \
        #                l[arrayIndexA[i]:arrayIndexA[i] + 1] + l[min(len(l) - 1, arrayIndexA[i] + 2):]
        # else:
        #     sortCSIa1P = list(sortCSIa1)
        #     sortCSIa1P.insert(arrayIndexA[i], allOpsIndexA[i])

        # if allOpsIndexB[i] == -1:
        #     sortCSIb1P = list(sortCSIb1)[:arrayIndexB[i]] + list(sortCSIb1)[arrayIndexB[i] + 1:]
        # elif allOpsIndexB[i] == -2:
        #     l = list(sortCSIb1)
        #     if arrayIndexB[i] + 2 >= len(l):
        #         continue
        #     sortCSIb1P = l[:arrayIndexB[i]] + l[arrayIndexB[i] + 1:arrayIndexB[i] + 2] + \
        #                l[arrayIndexB[i]:arrayIndexB[i] + 1] + l[min(len(l) - 1, arrayIndexB[i] + 2):]
        # else:
        #     sortCSIb1P = list(sortCSIb1)
        #     sortCSIb1P.insert(arrayIndexB[i], allOpsIndexB[i])

    print("sortCSIa1", len(sortCSIa1), list(sortCSIa1))
    print("sortCSIb1", len(sortCSIb1), list(sortCSIb1))
    print("sortCSIa1P", len(sortCSIa1P), list(sortCSIa1P))
    # print("sortCSIb1P", len(sortCSIb1P), sortCSIb1P)

    # 用a1P匹配ai，得到rule，再用rule对其a1P
    threshold = 2
    ruleStr1 = alignFloat(rule, sortCSIa1P, sortCSIa1, threshold)
    alignStr1 = genAlign(ruleStr1)
    print("ruleStr1", ruleStr1)
    ruleStr2 = alignFloat(rule, sortCSIa1P, sortCSIb1, threshold)
    alignStr2 = genAlign(ruleStr2)
    print("ruleStr2", ruleStr2)
    ruleStr3 = alignFloat(rule, sortCSIa1P, sortCSIe1, threshold)
    alignStr3 = genAlign(ruleStr3)
    print("ruleStr3", ruleStr3)
    ruleStr4 = alignFloat(rule, sortCSIa1P, sortNoise, threshold)
    alignStr4 = genAlign(ruleStr4)

    # 检错
    for i in range(min(len(ruleStr1), len(ruleStr2))):
        if ruleStr1[i] != ruleStr2[i]:
            if i >= len(sortCSIa1) or i >= len(sortCSIa1P): continue
            print("\033[0;35;40m", sortCSIa1[i], sortCSIb1[i], sortCSIa1P[i], "\033[0m")
            print(ruleStr1[i], abs(sortCSIa1[i] - sortCSIa1P[i]))
            print(ruleStr2[i], abs(sortCSIb1[i] - sortCSIa1P[i]))

    a_list = alignStr1
    b_list = alignStr2
    e_list = alignStr3
    n_list = alignStr4

    print("keys of a:", len(a_list), a_list)
    print("keys of b:", len(b_list), b_list)
    print("keys of e:", len(e_list), e_list)
    print("keys of n:", len(n_list), n_list)

    sum1 = min(len(a_list), len(b_list))
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(0, sum1):
        sum2 += (a_list[i] == b_list[i])
    for i in range(min(len(a_list), len(e_list))):
        sum3 += (a_list[i] == e_list[i])
    for i in range(min(len(a_list), len(n_list))):
        sum4 += (a_list[i] == n_list[i])

    print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    print("a-e", sum3, sum3 / sum1)
    print("a-n", sum4, sum4 / sum1)
    print("----------------------")
    originSum += sum1
    correctSum += sum2
    randomSum += sum3
    noiseSum += sum4

    # 编码密钥
    # char_weights = []
    # weights = Counter(a_list)  # 得到list中元素出现次数
    # for i in range(len(a_list)):
    #     char_weights.append((a_list[i], weights[a_list[i]]))
    # tree = HuffmanTree(char_weights)
    # tree.get_code()
    # HuffmanTree.codings += "\n"

    # for i in range(len(a_list)):
    #     codings += bin(a_list[i]) + "\n"

with open('../experiments/key.txt', 'a', ) as f:
    f.write(codings)

print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
