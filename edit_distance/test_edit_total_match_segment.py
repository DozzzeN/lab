import math
import os
import random

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import loadmat

from alignment import alignFloat, genAlign3, genLongestContinuous2


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


# 数组第二维的所有内容求和
def sumEachDim(list, index):
    res = 0
    for i in range(len(list[index])):
        res += list[index][i]
    return round(res, 8)


isShow = False
rawData = loadmat('../data/data_mobile_outdoor_1.mat')
if not os.path.exists('./figures/'):
    os.mkdir('./figures/')

# CSIa1OrigRaw = rawData['A'][:, 0]
# CSIb1OrigRaw = rawData['A'][:, 1]

# CSIa1Orig = []
# CSIb1Orig = []
# for i in range(500):
#     CSIa1Orig.append(CSIa1OrigRaw[i])
#     CSIb1Orig.append(CSIb1OrigRaw[i])
# for i in range(7000):
#     CSIa1Orig.append(CSIa1OrigRaw[i + 20000])
#     CSIb1Orig.append(CSIb1OrigRaw[i + 20000])
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
CSIi10rig = loadmat('../data/data_mobile_outdoor_2.mat')['A'][:, 0]

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)
CSIi10rig = np.array(CSIi10rig)

dataLen = len(CSIa1Orig)  # 6745

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

plt.close()
plt.figure()
plt.plot(range(len(CSIa1Orig[0:30])), CSIa1Orig[0:30], color="blue", linewidth=.5, label="CSIa1Orig raw")
plt.legend(loc='upper left')
plt.savefig('./figures/CSIa1Orig-raw.png')
if isShow:
    plt.show()
else:
    plt.close()

# 不进行平滑
# CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
# CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
# CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")
# CSIi10rig = smooth(CSIi10rig, window_len=15, window="flat")

# plt.figure()
# plt.plot(range(len(CSIa1Orig)), CSIa1Orig, color="cyan", linewidth=.5, label="CSIa1Orig smooth")
# plt.legend(loc='upper left')
# if isShow:
#     plt.show()
# else:
#     plt.close()

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()
CSIi1OrigBack = CSIi10rig.copy()

noise = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
noiseAdd = np.random.normal(loc=0, scale=10, size=dataLen)  ## Addition item normal distribution

sft = 2
intvl = 2 * sft + 1
keyLen = 128
segLen = 3
addNoise = True
rule = {'=': 0, '+': 1, '-': 3, '~': 2, '^': 3}
# rule = {'=': 0, '+': 1, '-': 1, '~': 1, '^': 1}

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

codings = ""
times = 10
maxDiffAB = 0
for staInd in range(0, times * intvl + 1, intvl):
    endInd = staInd + keyLen * intvl
    print("range:", staInd, endInd)
    if endInd > len(CSIa1Orig):
        break

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()
    CSIe1Orig = CSIe1OrigBack.copy()
    CSIi1Orig = CSIi1OrigBack.copy()

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
    tmpCSIi1 = CSIi1Orig[range(staInd, endInd, 1)]

    tmpNoise = noise[range(staInd, endInd, 1)]
    tmpNoiseAdd = noiseAdd[range(staInd, endInd, 1)]

    plt.figure()
    plt.plot(range(len(tmpCSIa1[0:10])), tmpCSIa1[0:10], color="black", linewidth=.5, label="tmpCSIa1 segment")
    plt.legend(loc='upper left')
    plt.savefig('./figures/tmpCSIa1-segment-' + str(staInd) + '.png')
    if isShow:
        plt.show()
    else:
        plt.close()

    tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency

    # linspace函数生成元素为50的等间隔数列，可以指定第三个参数为元素个数
    # signal.square返回周期性的方波波形
    tmpPulse = signal.square(
        2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

    if addNoise:
        # tmpCSIa1 = tmpPulse * (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
        # tmpCSIb1 = tmpPulse * (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
        # tmpCSIe1 = tmpPulse * (np.float_power(np.abs(tmpCSIe1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIe1))
        # tmpCSIi1 = tmpPulse * (np.float_power(np.abs(tmpCSIi1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIi1))
        tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpCSIa1), tmpNoise)
        tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpCSIb1), tmpNoise)
        tmpCSIe1 = tmpPulse * np.float_power(np.abs(tmpCSIe1), tmpNoise)
        tmpCSIi1 = tmpPulse * np.float_power(np.abs(tmpCSIi1), tmpNoise)
    else:
        tmpCSIa1 = tmpPulse * tmpCSIa1
        tmpCSIb1 = tmpPulse * tmpCSIb1
        tmpCSIe1 = tmpPulse * tmpCSIe1
        tmpCSIi1 = tmpPulse * tmpCSIi1

    plt.figure()
    plt.plot(range(len(tmpCSIa1[0:10])), tmpCSIa1[0:10], color="green", linewidth=.5,
             label="tmpCSIa1 segment add noise")
    plt.legend(loc='upper left')
    plt.savefig('./figures/tmpCSIa1-segment-add-noise-' + str(staInd) + '.png')
    if isShow:
        plt.show()
    else:
        plt.close()

    CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
    CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
    CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1
    CSIi1Orig[range(staInd, endInd, 1)] = tmpCSIi1

    permLen = len(range(staInd, endInd, intvl))
    origInd = np.array([xx for xx in range(staInd, endInd, intvl)])

    sortCSIa1 = np.zeros(permLen)
    sortCSIb1 = np.zeros(permLen)
    sortCSIe1 = np.zeros(permLen)
    sortNoise = np.zeros(permLen)
    sortCSIi1 = np.zeros(permLen)

    for ii in range(permLen):
        aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])  ## for non-permuted CSIa1

        for jj in range(permLen, permLen * 2):
            bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])

            CSIa1Tmp = CSIa1Orig[aIndVec]
            CSIb1Tmp = CSIb1Orig[bIndVec]
            CSIe1Tmp = CSIe1Orig[bIndVec]
            CSIi1Tmp = CSIi1Orig[bIndVec]
            noiseTmp = noise[aIndVec]

            sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
            sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # 只赋值一次
            sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
            sortCSIi1[jj - permLen] = np.mean(CSIi1Tmp)
            sortNoise[ii] = np.mean(noiseTmp)

    plt.figure()
    plt.plot(range(len(sortCSIa1[0:10])), sortCSIa1[0:10], color="blue", linewidth=.5, label="tmpCSIa1 segment mean")
    plt.legend(loc='upper left')
    plt.savefig('./figures/tmpCSIa1-segment-mean-' + str(staInd) + '.png')
    if isShow:
        plt.show()
    else:
        plt.close()

    # sortCSIa1是原始算法中排序前的数据
    # 防止对数的真数为0导致计算错误（不平滑的话没有这个问题）
    sortCSIa1 = np.log10(np.abs(sortCSIa1) + 0.01)
    sortCSIb1 = np.log10(np.abs(sortCSIb1) + 0.01)
    sortCSIe1 = np.log10(np.abs(sortCSIe1) + 0.01)
    sortNoise = np.log10(np.abs(sortNoise) + 0.01)
    sortCSIi1 = np.log10(np.abs(sortCSIi1) + 0.01)

    # 取原数据的一部分来reshape
    sortCSIa1Reshape = sortCSIa1[0:segLen * int(len(sortCSIa1) / segLen)]
    sortCSIb1Reshape = sortCSIb1[0:segLen * int(len(sortCSIb1) / segLen)]
    sortCSIe1Reshape = sortCSIe1[0:segLen * int(len(sortCSIe1) / segLen)]
    sortNoiseReshape = sortNoise[0:segLen * int(len(sortNoise) / segLen)]
    sortCSIi1Reshape = sortCSIi1[0:segLen * int(len(sortCSIi1) / segLen)]

    sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
    sortCSIe1Reshape = sortCSIe1Reshape.reshape(int(len(sortCSIe1Reshape) / segLen), segLen)
    sortNoiseReshape = sortNoiseReshape.reshape(int(len(sortNoiseReshape) / segLen), segLen)
    sortCSIi1Reshape = sortCSIi1Reshape.reshape(int(len(sortCSIi1Reshape) / segLen), segLen)

    sortCSIa1 = []
    sortCSIb1 = []
    sortCSIe1 = []
    sortNoise = []
    sortCSIi1 = []

    for i in range(len(sortCSIa1Reshape)):
        sortCSIa1.append(sumEachDim(sortCSIa1Reshape, i))
        sortCSIb1.append(sumEachDim(sortCSIb1Reshape, i))
        sortCSIe1.append(sumEachDim(sortCSIe1Reshape, i))
        sortNoise.append(sumEachDim(sortNoiseReshape, i))
        sortCSIi1.append(sumEachDim(sortCSIi1Reshape, i))

    plt.figure()
    plt.plot(range(len(sortCSIa1[0:10])), sortCSIa1[0:10], color="red", linewidth=.5, label="tmpCSIa1 segment log10")
    plt.legend(loc='upper left')
    plt.savefig('./figures/tmpCSIa1-segment-log10-' + str(staInd) + '.png')
    if isShow:
        plt.show()
    else:
        plt.close()

    shuffleArray = list(range(len(sortCSIa1)))
    CSIa1Back = []
    CSIb1Back = []
    CSIe1Back = []
    CSIn1Back = []
    random.shuffle(shuffleArray)
    for i in range(len(sortCSIa1)):
        CSIa1Back.append(sortCSIa1[shuffleArray[i]])
    for i in range(len(sortCSIb1)):
        CSIb1Back.append(sortCSIb1[shuffleArray[i]])
    for i in range(len(sortCSIe1)):
        CSIe1Back.append(sortCSIe1[shuffleArray[i]])
    for i in range(len(sortNoise)):
        CSIn1Back.append(sortNoise[shuffleArray[i]])
    sortCSIa1 = CSIa1Back
    sortCSIb1 = CSIb1Back
    sortCSIe1 = CSIe1Back
    sortNoise = CSIn1Back

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    diffAB = 0
    for i in range(len(sortCSIa1)):
        diffAB = max(diffAB, abs(sortCSIa1[i] - sortCSIb1[i]))
    print("\033[0;32;40mAB对应位置最大差距", diffAB, "\033[0m")
    maxDiffAB = max(maxDiffAB, diffAB)
    allDiffAB = 0
    for i in range(len(sortCSIa1)):
        for j in range(len(sortCSIb1)):
            allDiffAB = max(allDiffAB, abs(sortCSIa1[i] - sortCSIb1[j]))
    print("\033[0;32;40mAB所有的对应位置最大差距", allDiffAB, "\033[0m")

    opNums = int(len(sortCSIa1) / 2)
    index = random.sample(range(opNums), opNums)
    random.shuffle(sortCSIi1)

    sortCSIa1P = list(sortCSIa1)
    insertNum = 0
    deleteNum = 0
    updateNum = 0
    swapNum = 0

    opIndex = []
    editOps = []
    indices = []
    diffAB = 0.2
    insUpdBounds = 2
    for i in range(len(sortCSIa1)):
        editOps.append("=" + str(i))
    # sortCSIi1 = np.random.normal(loc=np.mean(sortCSIa1), scale=np.std(sortCSIa1, ddof=1), size=len(sortCSIa1))
    # sortCSIi1 = np.random.uniform(min(sortCSIa1), max(sortCSIa1), len(sortCSIa1))
    for i in range(opNums):
        # 按权重比例控制各种编辑的次数
        # flag = i % 12
        # index = random.randint(0, len(sortCSIa1P) - 1)
        # if flag in range(0, 6):
        #     insertIndex = random.randint(0, len(sortCSIi1) - 1)
        #     while math.fabs(sortCSIi1[insertIndex] - sortCSIa1P[index]) <= 2:
        #         insertIndex = random.randint(0, len(sortCSIi1) - 1)
        #     sortCSIa1P.insert(index, sortCSIi1[insertIndex])
        #     insertNum += 1
        # elif flag in range(6, 8):
        #     sortCSIa1P.remove(sortCSIa1P[index])
        #     deleteNum += 1
        # elif flag in range(8, 10):
        #     updateIndex = random.randint(0, len(sortCSIi1) - 1)
        #     while math.fabs(sortCSIa1P[index] - sortCSIi1[updateIndex]) <= 2:
        #         updateIndex = random.randint(0, len(sortCSIi1) - 1)
        #     sortCSIa1P[index] = sortCSIi1[updateIndex]
        #     updateNum += 1
        # elif flag in range(10, 12):
        #     swapIndex = random.randrange(0, len(sortCSIa1P) - 1, 2)
        #     tmp = sortCSIa1P[swapIndex]
        #     sortCSIa1P[swapIndex] = sortCSIa1P[swapIndex + 1]
        #     sortCSIa1P[swapIndex + 1] = tmp
        #     swapNum += 1

        flag = random.randint(0, 3)
        index = random.randint(0, len(sortCSIa1P) - 1)
        if flag == 0:
            insertIndex = random.randint(0, len(sortCSIi1) - 1)
            while math.fabs(sortCSIi1[insertIndex] - sortCSIa1P[index]) <= insUpdBounds:
                insertIndex = random.randint(0, len(sortCSIi1) - 1)
            sortCSIa1P.insert(index, sortCSIi1[insertIndex])
            editOps.insert(index, "+" + str(index))
            insertNum += 1
        elif flag == 1:
            sortCSIa1P.remove(sortCSIa1P[index])
            editOps[index] = "-" + str(index)
            deleteNum += 1
        elif flag == 2:
            updateIndex = random.randint(0, len(sortCSIi1) - 1)
            while math.fabs(sortCSIa1P[index] - sortCSIi1[updateIndex]) <= insUpdBounds:
                updateIndex = random.randint(0, len(sortCSIi1) - 1)
            sortCSIa1P[index] = sortCSIi1[updateIndex]
            editOps[index] = "~" + str(index)
            updateNum += 1
        elif flag == 3:
            swapIndex = random.randrange(0, len(sortCSIa1P) - 1, 2)
            tmp = sortCSIa1P[swapIndex]
            sortCSIa1P[swapIndex] = sortCSIa1P[swapIndex + 1]
            sortCSIa1P[swapIndex + 1] = tmp
            editOps[swapIndex] = "^" + str(index)
            swapNum += 1

    print("numbers of insert:", insertNum)
    print("numbers of delete:", deleteNum)
    print("numbers of update:", updateNum)
    print("numbers of swap:", swapNum)

    plt.figure()
    plt.plot(range(len(sortCSIa1P[0:10])), sortCSIa1P[0:10], color="red", linewidth=.5, label="sortCSIa1P segment edit")
    plt.legend(loc='upper left')
    if isShow:
        plt.show()
    else:
        plt.close()

    print("sortCSIa1P", len(sortCSIa1P), sortCSIa1P)
    print("sortCSIa1", len(sortCSIa1), list(sortCSIa1))
    print("sortCSIb1", len(sortCSIb1), list(sortCSIb1))
    print("sortCSIe1", len(sortCSIe1), list(sortCSIe1))
    # 编辑操作不好统计，因为删除的以后序列会打乱
    editOps.sort(key=lambda e: int(e[1:]))
    print("editOps", len(editOps), editOps)

    # 用a1P匹配ai，得到rule，再用rule对其a1P
    # threshold = 0.02
    threshold = diffAB
    # 只匹配相等的元素位置敌手的成功率很低，但加密强度不高
    # ruleStr1 = alignFloat(rule, sortCSIa1P, sortCSIa1, threshold)
    # alignStr1 = genAlign(ruleStr1)
    # print("ruleStr1", len(ruleStr1), ruleStr1)
    # ruleStr2 = alignFloat(rule, sortCSIa1P, sortCSIb1, threshold)
    # alignStr2 = genAlign(ruleStr2)
    # print("ruleStr2", len(ruleStr2), ruleStr2)
    # ruleStr3 = alignFloat(rule, sortCSIa1P, sortCSIe1, threshold)
    # alignStr3 = genAlign(ruleStr3)
    # print("ruleStr3", len(ruleStr3), ruleStr3)
    # ruleStr4 = alignFloat(rule, sortCSIa1P, sortNoise, threshold)
    # alignStr4 = genAlign(ruleStr4)

    # 匹配所有的编辑操作使得敌手的成功率变高
    # ruleStr1 = alignFloat(rule, sortCSIa1P, sortCSIa1, threshold)
    # alignStr1 = genAlign2(ruleStr1)
    # print("ruleStr1", len(ruleStr1), ruleStr1)
    # ruleStr2 = alignFloat(rule, sortCSIa1P, sortCSIb1, threshold)
    # alignStr2 = genAlign2(ruleStr2)
    # print("ruleStr2", len(ruleStr2), ruleStr2)
    # ruleStr3 = alignFloat(rule, sortCSIa1P, sortCSIe1, threshold)
    # alignStr3 = genAlign2(ruleStr3)
    # print("ruleStr3", len(ruleStr3), ruleStr3)
    # ruleStr4 = alignFloat(rule, sortCSIa1P, sortNoise, threshold)
    # alignStr4 = genAlign2(ruleStr4)

    # 匹配不相等的是在敌手成功率和加密强度间的折中
    ruleStr1 = alignFloat(rule, sortCSIa1P, sortCSIa1, threshold)
    alignStr1 = genAlign3(ruleStr1)
    print("ruleStr1", len(ruleStr1), ruleStr1)
    ruleStr2 = alignFloat(rule, sortCSIa1P, sortCSIb1, threshold)
    alignStr2 = genAlign3(ruleStr2)
    print("ruleStr2", len(ruleStr2), ruleStr2)
    ruleStr3 = alignFloat(rule, sortCSIa1P, sortCSIe1, threshold)
    alignStr3 = genAlign3(ruleStr3)
    print("ruleStr3", len(ruleStr3), ruleStr3)
    ruleStr4 = alignFloat(rule, sortCSIa1P, sortNoise, threshold)
    alignStr4 = genAlign3(ruleStr4)

    # 检错
    for i in range(min(len(ruleStr1), len(ruleStr2))):
        if ruleStr1[i] != ruleStr2[i]:
            indexAP = i
            indexA = i
            j = 0
            if j in range(i):
                if ruleStr1[j] == '-':
                    indexAP += 1
                    indexA -= 1
                if ruleStr1[j] == '+':
                    indexA += 1
                    indexAP -= 1
            if indexA + 1 >= len(sortCSIa1) or indexAP + 1 >= len(sortCSIa1P):
                continue
            if indexA - 1 < 0 or indexAP - 1 < 0:
                continue
            print("\033[0;30;42m", i, sortCSIa1[indexA], sortCSIb1[indexA], sortCSIa1P[indexAP], "\033[0m")
            print("\033[0;30;42m", sortCSIa1P[indexAP - 1], sortCSIa1P[indexAP], sortCSIa1P[indexAP + 1], "\033[0m")
            print("\033[0;30;42m", sortCSIa1[indexA - 1], sortCSIa1[indexA], sortCSIa1[indexA + 1], "\033[0m")
            print("\033[0;30;42m", sortCSIb1[indexA - 1], sortCSIb1[indexA], sortCSIb1[indexA + 1], "\033[0m")

    ruleStr1 = "".join(ruleStr1)
    ruleStr2 = "".join(ruleStr2)
    ruleStr3 = "".join(ruleStr3)
    ruleStr4 = "".join(ruleStr4)
    alignStr1 = genAlign3(ruleStr1)
    print("ruleStr1", len(ruleStr1), ruleStr1)
    alignStr2 = genAlign3(ruleStr2)
    print("ruleStr2", len(ruleStr2), ruleStr2)
    alignStr3 = genAlign3(ruleStr3)
    print("ruleStr3", len(ruleStr3), ruleStr3)
    alignStr4 = genAlign3(ruleStr4)

    a_list = alignStr1
    b_list = alignStr2
    e_list = alignStr3
    n_list = alignStr4

    print("keys of a:", len(a_list), a_list)
    print("keys of b:", len(b_list), b_list)
    print("keys of e:", len(e_list), e_list)
    print("keys of n:", len(n_list), n_list)

    print("longest numbers of a:", genLongestContinuous2(a_list))
    print("longest numbers of b:", genLongestContinuous2(b_list))
    print("longest numbers of e:", genLongestContinuous2(e_list))
    print("longest numbers of n:", genLongestContinuous2(n_list))

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

    if sum2 == sum1:
        print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b", sum2, sum2 / sum1, "\033[0m")
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

# with open('../experiments/key.txt', 'a', ) as f:
#     f.write(codings)

print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
print(maxDiffAB)
# segLen = 1
# a-b all 2620 / 2873 = 0.9119387399930386
# a-e all 12 / 2873 = 0.004176818656456666
# a-n all 15 / 2873 = 0.005221023320570832
# segLen = 2
# a-b all 1459 / 1547 = 0.9431157078215902
# a-e all 45 / 1547 = 0.029088558500323207
# a-n all 35 / 1547 = 0.02262443438914027
# segLen = 3
# a-b all 956 / 1015 = 0.941871921182266
# a-e all 51 / 1015 = 0.05024630541871921
# a-n all 50 / 1015 = 0.04926108374384237
# segLen = 4
# a-b all 802 / 809 = 0.9913473423980222
# a-e all 36 / 809 = 0.04449938195302843
# a-n all 38 / 809 = 0.04697156983930779
# segLen = 5
# a-b all 594 / 615 = 0.9658536585365853
# a-e all 31 / 615 = 0.05040650406504065
# a-n all 33 / 615 = 0.05365853658536585
# segLen = 6
# a-b all 559 / 559 = 1.0
# a-e all 28 / 559 = 0.05008944543828265
# a-n all 23 / 559 = 0.04114490161001789
# segLen = 7
# a-b all 446 / 446 = 1.0
# a-e all 33 / 446 = 0.07399103139013453
# a-n all 36 / 446 = 0.08071748878923767
# segLen = 8
# a-b all 420 / 422 = 0.995260663507109
# a-e all 43 / 422 = 0.1018957345971564
# a-n all 45 / 422 = 0.1066350710900474
# segLen = 10
# a-b all 319 / 319 = 1.0
# a-e all 37 / 319 = 0.11598746081504702
# a-n all 33 / 319 = 0.10344827586206896
