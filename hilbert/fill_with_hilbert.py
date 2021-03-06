import math
import sys

import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import loadmat


def findMinInterval(list):
    l = len(list)
    min_interval = sys.maxsize
    for i in range(l):
        for j in range(i + 1, l):
            interval = math.sqrt(math.pow(list[i][0] - list[j][0], 2) + math.pow(list[i][1] - list[j][1], 2))
            min_interval = min(min_interval, interval)
    return min_interval


def findMaxInterval(list):
    l = len(list)
    max_interval = 0
    for i in range(l):
        for j in range(i + 1, l):
            interval = math.sqrt(math.pow(list[i][0] - list[j][0], 2) + math.pow(list[i][1] - list[j][1], 2))
            max_interval = max(max_interval, interval)
    return max_interval


def findMinX(list):
    l = len(list)
    min_X = sys.maxsize
    for i in range(l):
        min_X = min(min_X, list[i][0])
    return min_X


def findMinY(list):
    l = len(list)
    min_X = sys.maxsize
    for i in range(l):
        min_X = min(min_X, list[i][1])
    return min_X


def findMaxX(list):
    l = len(list)
    min_X = 0
    for i in range(l):
        min_X = max(min_X, list[i][0])
    return min_X


def findMaxY(list):
    l = len(list)
    min_X = 0
    for i in range(l):
        min_X = max(min_X, list[i][1])
    return min_X


def listToHilbertCurveIndex(list):
    grid_size = findMinInterval(list) / 2 * math.sqrt(2)

    minX = findMinX(list)
    minY = findMinY(list)
    maxX = findMaxX(list)
    maxY = findMaxY(list)
    for i in range(len(list)):
        list[i][0] -= minX
        list[i][1] -= minY

    column_number = (maxX - minX) / grid_size
    row_number = (maxY - minY) / grid_size

    points = []

    for i in range(len(list)):
        indexX = int(list[i][0] / grid_size)
        indexY = int(list[i][1] / grid_size)
        points.append([indexX, indexY])

    total_grids = (int(column_number) + 1) * (int(row_number) + 1)
    p = int(math.log10(total_grids) / math.log10(2))

    print(p, grid_size)

    n = 2
    hilbert_curve = HilbertCurve(p, n)
    distances = hilbert_curve.distances_from_points(points, match_type=True)
    return distances


def cut(list):
    for i in range(len(list)):
        # list[i] = str(list[i])[0:int(len(str(list[i])) / 2)]
        list[i] = str(list[i])[0:2]
        list[i] = int(list[i])
    return list


def listToHilbertCurveIndexWithFixedParams(list, p, grid_size):
    minX = findMinX(list)
    minY = findMinY(list)
    for i in range(len(list)):
        list[i][0] -= minX
        list[i][1] -= minY

    points = []

    for i in range(len(list)):
        indexX = int(list[i][0] / grid_size)
        indexY = int(list[i][1] / grid_size)
        points.append([indexX, indexY])

    n = 2
    hilbert_curve = HilbertCurve(p, n)
    distances = hilbert_curve.distances_from_points(points, match_type=True)
    return distances


def smooth(x, window_len=11, window='hanning'):
    # ndim?????????????????????
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser'")

    # np.r_????????????????????????????????????????????????????????????????????????
    # ??????[????????????:????????????:????????????]
    # ???????????????????????????????????????
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        # ?????????float?????????window_len???1.?????????
        w = np.ones(window_len, 'd')
    elif window == 'kaiser':
        beta = 5
        w = eval('np.' + window + '(window_len, beta)')
    else:
        w = eval('np.' + window + '(window_len)')

    # ??????????????????
    y = np.convolve(w / w.sum(), s, mode='valid')  # 6759
    return y


# ????????????????????????????????????
def sumEachDim(list, index):
    res = 0
    for i in range(len(list[index])):
        res += (list[index][i][0] + list[index][i][1])
        # res += (list[index][i][0] * list[index][i][1])
    return round(res, 8)


def projection(l1, l2, p):
    v1 = [p[0] - l1[0], p[1] - l1[1]]
    v2 = [l2[0] - l1[0], l2[1] - l1[1]]
    v1v2 = v1[0] * v2[0] + v1[1] * v2[1]
    k = v1v2 / (math.pow(l2[0] - l1[0], 2) + math.pow(l2[1] - l1[1], 2))
    p0 = [l1[0] + k * (l2[0] - l1[0]), l1[1] + k * (l2[1] - l1[1])]
    return p0


l1 = [1, 0]
l2 = [0, 1]
rawData = loadmat('../data/data_mobile_indoor_1.mat')

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

# rawData = loadmat('../data/data_static_indoor_1.mat')
#
# CSIa1Orig = rawData['A'][:, 0]
# CSIb1Orig = rawData['A'][:, 1]
#
# CSIa1Orig = CSIa1Orig[25000:30000]
# CSIb1Orig = CSIb1Orig[25000:30000]

# CSIa1OrigRaw = rawData['A'][:, 0]
# CSIb1OrigRaw = rawData['A'][:, 1]
#
# CSIa1Orig = []
# CSIb1Orig = []
# for i in range(2000):
#     CSIa1Orig.append(CSIa1OrigRaw[i])
#     CSIb1Orig.append(CSIb1OrigRaw[i])
# for i in range(5000):
#     CSIa1Orig.append(CSIa1OrigRaw[i + 20000])
#     CSIb1Orig.append(CSIb1OrigRaw[i + 20000])

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)

# plt.figure()
# plt.plot(CSIa1Orig, color="red", linewidth=.05, label="a")
# # plt.plot(CSIb1Orig, color="blue", linewidth=.05, label="b")
# plt.legend(loc='upper left')
# plt.show()

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

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

codings = ""
# for ii in range(0, 5):
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

    # linspace?????????????????????50???????????????????????????????????????????????????????????????
    # signal.square??????????????????????????????
    tmpPulse = signal.square(
        2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

    if addNoise:
        tmpCSIa1 = tmpPulse * (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
        tmpCSIb1 = tmpPulse * (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
        tmpCSIe1 = tmpPulse * (np.float_power(np.abs(tmpCSIe1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIe1))
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
            sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # ???????????????
            sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
            sortNoise[ii] = np.mean(noiseTmp)

    # ?????????
    # _max = max(max(sortCSIa1), max(sortCSIb1), max(sortCSIe1), max(sortNoise))
    # _min = min(min(sortCSIa1), min(sortCSIb1), min(sortCSIe1), min(sortNoise))

    # sortCSIa1 = sortCSIa1 / (_max - _min) - _min / (_max - _min)
    # sortCSIb1 = sortCSIb1 / (_max - _min) - _min / (_max - _min)
    # sortCSIe1 = sortCSIe1 / (_max - _min) - _min / (_max - _min)
    # sortNoise = sortNoise / (_max - _min) - _min / (_max - _min)

    # sortCSIa1????????????????????????????????????
    sortCSIa1 = np.log10(np.abs(sortCSIa1))
    sortCSIb1 = np.log10(np.abs(sortCSIb1))
    sortCSIe1 = np.log10(np.abs(sortCSIe1))
    sortNoise = np.log10(np.abs(sortNoise))

    # ??????????????????????????????????????????????????????
    # ??????????????????param??????
    param = 0
    step = int(math.pow(2, param))
    sortCSIa1 = sortCSIa1.reshape(int(len(sortCSIa1) / step / 2), 2)
    sortCSIb1 = sortCSIb1.reshape(int(len(sortCSIb1) / step / 2), 2)
    sortCSIe1 = sortCSIe1.reshape(int(len(sortCSIe1) / step / 2), 2)
    sortNoise = sortNoise.reshape(int(len(sortNoise) / step / 2), 2)

    plt.figure()
    plt.plot(sortCSIa1[:, 0], sortCSIa1[:, 1], color="red", linewidth=1, label="a")
    # plt.plot(CSIb1Orig, color="blue", linewidth=.05, label="b")
    plt.legend(loc='upper left')
    # plt.show()

    # ?????????????????????
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    projCSIa1XY = []
    projCSIb1XY = []
    projCSIe1XY = []
    projCSIn1XY = []

    for i in range(len(sortCSIa1)):
        # projCSIa1XY.append(projection(l1, l2, sortCSIa1[i]))
        # projCSIb1XY.append(projection(l1, l2, sortCSIb1[i]))
        # projCSIe1XY.append(projection(l1, l2, sortCSIe1[i]))
        # projCSIn1XY.append(projection(l1, l2, sortNoise[i]))
        projCSIa1XY.append([sortCSIa1[i][0], sortCSIa1[i][1]])
        projCSIb1XY.append([sortCSIb1[i][0], sortCSIb1[i][1]])
        projCSIe1XY.append([sortCSIe1[i][0], sortCSIe1[i][1]])
        projCSIn1XY.append([sortNoise[i][0], sortNoise[i][1]])

    a_list = listToHilbertCurveIndexWithFixedParams(projCSIa1XY, 20, 10)
    b_list = listToHilbertCurveIndexWithFixedParams(projCSIb1XY, 20, 10)
    e_list = listToHilbertCurveIndexWithFixedParams(projCSIe1XY, 20, 10)
    n_list = listToHilbertCurveIndexWithFixedParams(projCSIn1XY, 20, 10)
    # a_list = listToHilbertCurveIndex(projCSIa1XY)
    # b_list = listToHilbertCurveIndex(projCSIb1XY)
    # e_list = listToHilbertCurveIndex(projCSIe1XY)
    # n_list = listToHilbertCurveIndex(projCSIn1XY)

    # a_list = cut(a_list)
    # b_list = cut(b_list)
    # e_list = cut(e_list)
    # n_list = cut(n_list)

    print("keys of a:", a_list)
    print("keys of b:", b_list)
    print("keys of e:", e_list)
    print("keys of n:", n_list)

    sum1 = len(a_list)
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(0, sum1):
        sum2 += (a_list[i] - b_list[i] == 0)
        sum3 += (a_list[i] - e_list[i] == 0)
        sum4 += (a_list[i] - n_list[i] == 0)

    print("a-b", sum2 / sum1)
    print("a-e", sum3 / sum1)
    print("a-n", sum4 / sum1)
    print("----------------------")
    originSum += sum1
    correctSum += sum2
    randomSum += sum3
    noiseSum += sum4

    # ????????????
    # char_weights = []
    # weights = Counter(a_list)  # ??????list?????????????????????
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
