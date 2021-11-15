import math
import os
import random
import shutil
import sys

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon


def genRandomStep(len, lowBound, highBound):
    length = 0
    randomStep = []
    # 少于三则无法分，因为至少要划分出一个三角形
    while len - length >= lowBound:
        step = random.randint(lowBound, highBound)
        randomStep.append(step)
        length += step
    return randomStep


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


sortCSIa1 = [1.69158026e+046, -8.52511024e-004, -1.90920077e+034, 3.26485671e+079,
             -3.78190803e+010, -3.68405556e+050, 1.19696497e+050, 1.08132325e+038,
             -5.26179743e+025, -2.16697913e+050, 1.56639967e+005, -8.33889963e+054,
             -3.17970104e+013, 4.39000268e+023, -4.76385956e+004, -9.01687162e+029,
             1.53963915e+027, 2.36209833e+055, -1.00988976e+014, 2.84052243e+013,
             -7.39535546e+024, -5.29708652e+025, -1.24939867e+019, 1.40425933e+127,
             -9.98080136e+038, -4.59125081e+072, 2.22089983e+046, -2.82311752e+018,
             -2.06820473e+033, 1.10743542e+061, 2.80211730e+017, -2.70011116e+041,
             3.47539815e+038, -8.70761702e+022, -6.37797325e+005, 5.00272494e+037,
             -4.83331524e+116, -3.02465596e+034, 1.11596079e+002, -7.39430045e+047,
             -1.14476270e-002, 9.22685235e+059, 1.74176021e+096, -1.67206823e+049,
             2.95301623e+039, 2.08031829e+041, -7.88688553e+014, -9.22272876e+068,
             3.85323446e+018, 1.18979457e+015, -6.43741811e+048, 1.54115618e+010,
             -4.90337738e+045, -1.72228665e+083, 8.56562949e+212, -4.15972794e-001,
             -2.64977025e+083, 2.62980620e+044, 3.55341706e+031, -9.56713473e+084,
             -1.37586148e+063, 2.72466594e-008, -3.04284450e+019, -2.03167374e+033,
             4.45285125e+039, -2.11097932e+020, -5.70974975e+010, 3.31811080e+075,
             -7.55533145e+014, -2.55180377e+040, 2.97841295e+011, 1.28459841e+023,
             -3.36876941e+019, 1.24577143e+016, 6.58686045e+055, -1.31043654e+056,
             1.28690007e-009, 9.08688792e+002, -1.83662073e+068, -4.30248304e+013,
             5.22967574e+015, -3.92990747e+021, -7.23418155e+036, 1.04667614e+038,
             -7.38052088e+033, -1.13885049e+004, 7.37076047e+019, -4.54143087e+016,
             -1.12295213e+035, 5.58125192e+068, 3.33657498e+006, -4.22295297e+077,
             3.04622515e+069, 7.88139990e+038, -2.57667855e+011, -1.53902080e+023,
             1.77339219e+086, -1.15434981e+052, -1.78033566e+071, 5.72575344e+019,
             -3.47958113e+000, -8.40334892e+048, 2.76485544e+092, 1.02102189e+075,
             -2.10469662e+042, 1.37468990e+067, -8.30792744e+052, -2.97540012e+065,
             3.45985596e+006, 1.22246803e+035, -7.72751216e-009, 4.59658261e+026,
             6.34445134e+049, -1.90222638e+035, -3.04918891e+051, 5.82961425e+006,
             5.44544576e+057, 1.76879784e+056, 2.63964476e+012, -1.33334284e+064,
             2.81811078e+004, 4.50890639e+072, -2.82487029e+109, -1.43386909e+028,
             1.38024430e+040, 2.44602089e+039, -1.35691993e+014, 1.24317941e+050]
sortCSIb1 = [5.11320538e+047, -8.20667496e-004, -1.81383023e+035, 7.31414788e+078,
             -2.70988246e+010, -8.00638587e+049, 3.45878718e+049, 6.34950982e+037,
             -3.38560471e+025, -9.42653366e+049, 1.48296945e+005, -3.74950881e+054,
             -2.29865457e+013, 1.91614802e+023, -3.99930461e+004, -4.01736648e+029,
             8.58231060e+026, 6.11095149e+054, -6.18411623e+013, 1.58819388e+013,
             -2.70493109e+024, -2.00311390e+025, -4.51016019e+018, 1.09031453e+123,
             -1.57155641e+038, -5.78561035e+071, 1.76143076e+046, -4.15312869e+018,
             -1.40753019e+033, 7.69082267e+060, 2.08870044e+017, -4.22939528e+041,
             3.10014203e+038, -5.74754314e+022, -5.92541106e+005, 2.96573221e+037,
             -1.67685179e+116, -3.07669606e+034, 1.11796201e+002, -2.76632732e+048,
             -1.14509147e-002, 4.06375116e+060, 1.38853512e+097, -9.34009715e+048,
             1.43851186e+039, 2.11867988e+041, -1.89685751e+015, -1.93395895e+070,
             8.73438133e+018, 1.26515361e+015, -4.15198754e+048, 1.60311752e+010,
             -4.39195245e+045, -1.41610372e+083, 5.17747718e+212, -4.30178785e-001,
             -1.57172140e+084, 9.19860297e+044, 7.86414702e+031, -3.56165773e+085,
             -2.50015112e+063, 2.43383831e-008, -5.23206006e+019, -3.79070040e+033,
             6.52073675e+039, -2.58170441e+020, -6.51102922e+010, 1.34571032e+076,
             -8.37791699e+014, -3.72312765e+040, 3.22512786e+011, 1.49194387e+023,
             -3.83840897e+019, 1.31663735e+016, 1.58387079e+056, -4.53518733e+056,
             1.09150604e-009, 9.68226310e+002, -1.97342631e+069, -7.20882169e+013,
             6.85286369e+015, -4.56012979e+021, -6.58119824e+036, 1.19489093e+038,
             -7.50401813e+033, -1.16602166e+004, 8.87746628e+019, -4.80161112e+016,
             -9.33752880e+034, 4.73181128e+068, 3.36725211e+006, -1.06007379e+078,
             5.71170237e+069, 1.13289835e+039, -2.95092850e+011, -1.66444965e+023,
             1.12507946e+086, -7.56031717e+051, -1.22363721e+071, 5.78099510e+019,
             -3.55548935e+000, -1.74765533e+049, 2.50159457e+093, 4.89572744e+075,
             -4.59635857e+042, 2.11253966e+067, -7.28528750e+052, -2.53222877e+065,
             3.31232332e+006, 1.24347985e+035, -7.25245660e-009, 5.95267364e+026,
             1.17885768e+050, -3.64564522e+035, -9.25472631e+051, 6.51767886e+006,
             5.58711817e+057, 1.35150913e+056, 2.55951506e+012, -6.80041985e+063,
             2.60744824e+004, 5.78347087e+071, -1.27714402e+108, -7.41549843e+027,
             4.82399264e+039, 1.27463213e+039, -9.73675058e+013, 9.77800087e+049]
sortCSIe1 = [3.15129700e+055, -5.06172296e-004, -7.74002367e+043, 5.58802680e+100,
             -9.52807605e+012, -1.80640094e+065, 4.24872492e+066, 1.15168761e+052,
             -1.87066002e+036, -1.44932010e+064, 4.72768024e+006, -3.64963716e+070,
             -1.01117970e+018, 1.46238509e+035, -1.82427131e+007, -1.29232845e+039,
             1.75846724e+035, 1.16548929e+076, -5.29927430e+021, 4.01918072e+020,
             -3.57340450e+033, -1.58592480e+033, -1.34586815e+025, 2.37287600e+162,
             -3.30521842e+050, -3.25647003e+088, 1.19629024e+056, -4.52808852e+021,
             -3.00265890e+043, 7.02310134e+092, 6.32163220e+026, -2.77364129e+063,
             5.26586368e+059, -1.26964653e+037, -1.97357620e+010, 3.73672209e+061,
             -1.46832510e+167, -3.82662665e+043, 1.25632423e+003, -3.88491464e+060,
             -1.19938171e-002, 1.61994316e+074, 1.45250736e+120, -2.23269847e+067,
             5.80346897e+052, 2.88544111e+051, -6.91613668e+016, -1.73537739e+080,
             6.82811641e+023, 3.23830739e+021, -8.13902922e+064, 3.34558785e+013,
             -8.85312257e+060, -1.51893864e+111, 6.62175548e+292, -1.36371929e+000,
             -1.51142301e+113, 1.68672323e+061, 3.24120629e+046, -5.18595568e+127,
             -1.94433926e+093, 2.04818942e-010, -1.60794033e+027, -1.65578129e+046,
             2.53208439e+054, -7.58156408e+027, -7.16101654e+013, 1.86407813e+101,
             -1.03311674e+021, -3.33952483e+055, 1.36812271e+015, 1.34970075e+030,
             -3.24888378e+027, 5.30782762e+022, 3.21285332e+080, -2.17656927e+075,
             7.18487011e-012, 8.46340367e+003, -3.99802316e+100, -7.58462682e+020,
             9.05236279e+023, -9.92537532e+031, -7.32354667e+052, 2.49763710e+052,
             -1.47496263e+046, -1.93252096e+005, 2.13975928e+026, -3.25544029e+021,
             -6.11945923e+043, 9.07647245e+089, 5.24418375e+009, -3.63753275e+104,
             1.13890037e+095, 4.90895405e+050, -1.46779294e+015, -4.28138944e+029,
             2.28473431e+108, -1.63727077e+065, -4.23268599e+093, 2.00130090e+026,
             -1.71819688e+001, -2.22560682e+065, 3.14765812e+125, 2.29117710e+103,
             -5.12756889e+058, 4.48770036e+092, -1.75141815e+076, -1.65689252e+090,
             1.41936982e+009, 1.91149764e+046, -1.62399728e-010, 1.47723075e+035,
             1.54098472e+068, -1.38173098e+048, -2.10252369e+071, 2.05439041e+008,
             1.06140604e+069, 1.07297183e+066, 1.04699185e+016, -8.11670398e+083,
             6.86803360e+005, 2.60975155e+096, -5.70206044e+144, -4.74379507e+038,
             4.73969791e+055, 3.96746773e+051, -6.67170697e+016, 5.18039082e+056]
sortNoise = [-1.11468530e+00, -1.98689072e+00, -9.42213928e-01, -7.36188392e-01,
             -1.19215235e+00, -8.59282522e-01, -1.01126663e+00, -1.22946598e+00,
             -7.83748685e-01, -1.07001329e+00, -1.37513462e+00, -5.71806310e-01,
             -1.10681373e+00, -1.64391326e+00, -8.58404915e-01, -3.29664207e-01,
             -1.64441677e+00, -1.04790160e+00, -6.18374662e-01, -1.27927429e+00,
             -1.20636040e+00, -1.38115215e+00, -6.51404731e-01, -1.71313993e-01,
             -1.18731433e+00, -6.09902043e-01, -4.34335923e-01, -7.40292023e-01,
             -1.20360883e+00, -7.97315669e-01, -9.46779883e-01, -5.02877034e-01,
             -1.66395045e+00, -4.35948694e-01, -1.51041533e+00, -5.14807024e-01,
             -9.06234720e-01, -9.51382423e-01, -1.04298462e+00, -2.20618811e-01,
             -1.60534903e+00, -1.15514206e+00, -1.64482110e+00, -8.65877092e-01,
             -5.45591203e-01, -8.53309941e-01, -1.01670711e+00, -9.41361005e-01,
             -1.06454006e+00, -7.00081167e-01, -1.31132584e+00, -9.42967361e-01,
             -1.12458574e+00, -6.69554469e-01, -8.82273548e-01, -1.25938695e+00,
             -1.05479728e+00, -5.99101434e-01, -1.11993875e+00, 1.54099733e-03,
             -5.49168046e-01, -1.73758866e+00, -3.87251194e-01, -3.71560676e-01,
             -1.28737803e-01, -1.03628119e+00, -1.52396554e+00, -9.32312355e-01,
             -6.23717480e-01, -8.33182818e-01, -1.13337461e+00, -9.68650317e-01,
             -9.82199053e-01, -7.42556691e-01, -1.44913116e+00, -9.31808661e-01,
             -1.77942807e+00, -1.22293838e+00, -9.92495129e-01, -1.35522547e+00,
             -1.77014766e+00, -1.01057181e+00, -7.56641354e-01, -5.37662951e-01,
             -7.07691561e-01, -9.10213312e-01, -7.77149213e-01, -7.01822080e-01,
             -8.98210484e-01, -1.38856428e+00, -9.37680300e-01, -6.15841089e-01,
             -1.46535034e+00, -1.02830622e+00, -1.46096360e+00, -4.16149111e-01,
             -6.58908770e-01, -7.80466836e-01, -1.01400083e+00, -1.19040908e+00,
             -1.60278200e+00, -1.17995554e+00, -1.22898157e+00, -4.78476967e-01,
             -1.05295875e+00, -8.38380637e-01, -5.89996539e-01, -7.35350297e-01,
             -1.52324709e+00, -7.54068305e-01, -2.00614000e+00, -1.15494449e+00,
             -1.87504797e-01, -4.16522902e-01, -4.51844476e-01, -1.51382989e+00,
             -8.15132059e-01, -5.97119385e-01, -1.25624808e+00, -2.15487136e-01,
             -1.35835905e+00, -1.02618283e+00, -8.31232167e-01, -7.11046971e-01,
             -1.03983454e+00, -1.18589832e+00, -9.22387861e-01, -9.37123270e-01]


# 优化算法的计算复杂度O((n+m)log(n+m)),暂时使用穷举
# 标准单向hausdorff距离
# h(A,B) = max  min ||ai-bj||
#          ai∈A bj∈B
def standard_hd(x, y):
    h1 = 0
    for xi in x:
        shortest = sys.maxsize
        for yi in y:
            d = round(math.pow(xi[0] - yi[0], 2) + math.pow(xi[1] - yi[1], 2), 10)
            if d < shortest:
                shortest = d
        if shortest > h1:
            h1 = shortest

    h2 = 0
    for xi in y:
        shortest = sys.maxsize
        for yi in x:
            d = round(math.pow(xi[0] - yi[0], 2) + math.pow(xi[1] - yi[1], 2), 10)
            if d < shortest:
                shortest = d
        if shortest > h2:
            h2 = shortest
    return max(h1, h2)


# 平均单向hausdorff距离的效果差些
def average_hd(x, y):
    h = 0
    for xi in x:
        shortest = sys.maxsize
        for yi in y:
            d = round(math.pow(xi[0] - yi[0], 2) + math.pow(xi[1] - yi[1], 2), 10)
            if d < shortest:
                shortest = d
        h += shortest
    return h / len(x)


# 添加数组头元素以构成封闭的多边形
def makePolygon(list):
    listPolygon = []
    for i in range(len(list)):
        listTmp = []
        for j in range(len(list[i])):
            listTmp.append(list[i][j])
        listTmp.append(list[i][0])
        listPolygon.append(listTmp)
    return listPolygon


# 将三维数组转为一维数组
def toOneDim(list):
    oneDim = []
    for i in range(len(list)):
        tmp = 0
        for j in range(len(list[i])):
            tmp += list[i][j][0] + list[i][j][1]
        oneDim.append(tmp)
    return oneDim


# 数组第二维的所有内容求和
def sumEachDim(list, index):
    res = 0
    for i in range(len(list[index])):
        res += list[index][i][0] + list[index][i][1]
    return res


# 归一化
# _max = max(max(sortCSIa1), max(sortCSIb1), max(sortCSIe1), max(sortNoise))
# _min = min(min(sortCSIa1), min(sortCSIb1), min(sortCSIe1), min(sortNoise))

# sortCSIa1 = sortCSIa1 / (_max - _min) - _min / (_max - _min)
# sortCSIb1 = sortCSIb1 / (_max - _min) - _min / (_max - _min)
# sortCSIe1 = sortCSIe1 / (_max - _min) - _min / (_max - _min)
# sortNoise = sortNoise / (_max - _min) - _min / (_max - _min)

# sortCSIa1是原始算法中排序前的数据
sortCSIa1 = np.log10(np.abs(sortCSIa1)) * 1 / 2 - 10
sortCSIb1 = np.log10(np.abs(sortCSIb1))
sortCSIe1 = np.log10(np.abs(sortCSIe1)) * 2 + 10
sortNoise = np.log10(np.abs(sortNoise)) * 2 + 10

_max = max(sortCSIa1)
_min = min(sortCSIa1)

# 形成三维数组，其中第三维是一对坐标值
# 数组的长度由param调节
param = 2
step = int(math.pow(2, param))
sortCSIa1Copy = sortCSIa1.copy()
sortCSIb1Copy = sortCSIb1.copy()
sortCSIe1Copy = sortCSIe1.copy()
sortCSIn1Copy = sortNoise.copy()

sortCSIa1Copy = sortCSIa1Copy.reshape(int(len(sortCSIa1) / 2), 2)
sortCSIb1Copy = sortCSIb1Copy.reshape(int(len(sortCSIb1) / 2), 2)
sortCSIe1Copy = sortCSIe1Copy.reshape(int(len(sortCSIe1) / 2), 2)
sortCSIn1Copy = sortCSIn1Copy.reshape(int(len(sortNoise) / 2), 2)

sortCSIa1Split = []
sortCSIb1Split = []
sortCSIe1Split = []
sortNoiseSplit = []

randomStep = genRandomStep(int(len(sortCSIa1) / 2), step, step)
print("randomStep", randomStep)
startIndex = 0

del_file('./figures/')

for i in range(len(randomStep)):
    if not os.path.exists('./figures/' + "not_convex"):
        os.mkdir('./figures/' + "not_convex")

    # 由于随机产生step的算法不一定刚好满足step之和等于keyLen/2，故在每次复制值的时候需要判断
    if startIndex >= len(sortCSIa1Copy) or len(sortCSIa1Copy) - startIndex < 3:
        break
    sortCSIa1Split.append(sortCSIa1Copy[startIndex:startIndex + randomStep[i]])
    sortCSIb1Split.append(sortCSIb1Copy[startIndex:startIndex + randomStep[i]])
    sortCSIe1Split.append(sortCSIe1Copy[startIndex:startIndex + randomStep[i]])
    sortNoiseSplit.append(sortCSIn1Copy[startIndex:startIndex + randomStep[i]])
    startIndex = startIndex + randomStep[i]

sortCSIa1 = sortCSIa1Split
sortCSIb1 = sortCSIb1Split
sortCSIe1 = sortCSIe1Split
sortNoise = sortNoiseSplit

CSIa1Back = sortCSIa1
CSIb1Back = sortCSIb1
CSIe1Back = sortCSIe1
CSIn1Back = sortNoise

# for i in range(len(CSIa1Back)):
#     is_in = Polygon(CSIa1Back[i]).intersects(Polygon(sortCSIb1[i]))
#     while is_in is True:
#         for j in range(len(CSIa1Back[i])):
#             CSIa1Back[i][j] = [random.uniform(0, np.log10(np.abs(_max - _min))),
#                                random.uniform(0, np.log10(np.abs(_max - _min)))]
#         is_in = Polygon(CSIa1Back[i]).intersects(Polygon(sortCSIb1[i]))

# 凸包检查
# for i in range(len(CSIa1Back)):
#     CSIa1Back[i] = list(Polygon(CSIa1Back[i]).convex_hull.exterior.coords)[0:-1]
#     CSIb1Back[i] = list(Polygon(CSIb1Back[i]).convex_hull.exterior.coords)[0:-1]
#     CSIe1Back[i] = list(Polygon(CSIe1Back[i]).convex_hull.exterior.coords)[0:-1]
#     CSIn1Back[i] = list(Polygon(CSIn1Back[i]).convex_hull.exterior.coords)[0:-1]

# 降维以用于后续的排序
oneDimCSIa1 = toOneDim(CSIa1Back)
oneDimCSIb1 = toOneDim(CSIb1Back)
oneDimCSIe1 = toOneDim(CSIe1Back)
oneDimCSIn1 = toOneDim(CSIn1Back)

# # 初始化一维数组，作为基准数组
# sortCSIa1Back = np.sort(oneDimCSIa1, axis=0)
# sortCSIb1Back = np.sort(oneDimCSIb1, axis=0)
# sortCSIe1Back = np.sort(oneDimCSIe1, axis=0)
# sortCSIn1Back = np.sort(oneDimCSIn1, axis=0)

# 计算hd距离和多边形的顺序无关，可以任意洗牌
# CSIa1Back = [[] for _ in range(len(sortCSIa1))]
# CSIb1Back = [[] for _ in range(len(sortCSIb1))]
# CSIe1Back = [[] for _ in range(len(sortCSIe1))]
# CSIn1Back = [[] for _ in range(len(sortNoise))]
#
# rand_out_polygon = list(range(len(sortCSIa1)))
# rand_in_polygon = list(range(step))
#
# random.shuffle(rand_out_polygon)
# random.shuffle(rand_in_polygon)
# for i in range(len(sortCSIa1)):
#     for j in range(step):
#         CSIa1Back[i].append(sortCSIa1[rand_out_polygon[i]][rand_in_polygon[j]])
#
# random.shuffle(rand_out_polygon)
# random.shuffle(rand_in_polygon)
# for i in range(len(sortCSIb1)):
#     for j in range(step):
#         CSIb1Back[i].append(sortCSIb1[rand_out_polygon[i]][rand_in_polygon[j]])
#
# random.shuffle(rand_out_polygon)
# random.shuffle(rand_in_polygon)
# for i in range(len(sortCSIe1)):
#     for j in range(step):
#         CSIe1Back[i].append(sortCSIe1[rand_out_polygon[i]][rand_in_polygon[j]])
#
# random.shuffle(rand_out_polygon)
# random.shuffle(rand_in_polygon)
# for i in range(len(sortNoise)):
#     for j in range(step):
#         CSIn1Back[i].append(sortNoise[rand_out_polygon[i]][rand_in_polygon[j]])

# 在数组a后面加上a[0]使之成为一个首尾封闭的多边形
sortCSIa1Add = makePolygon(CSIa1Back)
sortCSIb1Add = makePolygon(CSIb1Back)
sortCSIe1Add = makePolygon(CSIe1Back)
sortCSIn1Add = makePolygon(CSIn1Back)

# 初始化各个计算出的hd值
ab_max = 0
ae_max = 0
an_max = 0

# 最后各自的密钥
a_list = []
b_list = []
e_list = []
n_list = []

for i in range(len(sortCSIa1Add)):
    xa, ya = zip(*sortCSIa1Add[i])
    xb, yb = zip(*sortCSIb1Add[i])
    xe, ye = zip(*sortCSIe1Add[i])
    xn, yn = zip(*sortCSIn1Add[i])
    plt.figure()
    plt.plot(xa, ya, color="red", linewidth=2.5, label="a" + str(i))
    plt.plot(xb, yb, color="blue", linewidth=2.5, label="b" + str(i))
    plt.plot(xe, ye, color="black", linewidth=2.5, label="e" + str(i))
    # plt.plot(xn, yn, color="yellow", linewidth=2.5, label="n") # 数量级差别太大，不方便显示
    plt.legend(loc='upper left')
    plt.savefig('./figures/not_convex/' + str(i) + '.png')
    plt.close()
    # plt.show()

for i in range(len(CSIa1Back)):
    ab_hd = sys.maxsize
    ae_hd = sys.maxsize
    an_hd = sys.maxsize

    ab_index = 0
    ae_index = 0
    an_index = 0
    for j in range(len(CSIa1Back)):
        # 整体计算两个集合中每个多边形的hd值，取最匹配的（hd距离最接近的两个多边形）
        ab_d = Polygon(CSIa1Back[i]).hausdorff_distance(Polygon(CSIb1Back[j]))
        ae_d = Polygon(CSIa1Back[i]).hausdorff_distance(Polygon(CSIe1Back[j]))
        an_d = Polygon(CSIa1Back[i]).hausdorff_distance(Polygon(CSIn1Back[j]))
        # ab_d = average_hd(CSIa1Back[i], CSIb1Back[j])
        # ae_d = average_hd(CSIa1Back[i], CSIe1Back[j])
        # an_d = average_hd(CSIa1Back[i], CSIn1Back[j])
        if ab_d < ab_hd:
            ab_hd = ab_d
            ab_index = j
        if ae_d < ae_hd:
            ae_hd = ae_d
            ae_index = j
        if an_d < an_hd:
            an_hd = an_d
            an_index = j
    if ae_index != ab_index:
        print("not equal ae_index", ae_index, "ae_hd", ae_hd)
        print("not equal ab_index", ab_index, "ab_hd", ab_hd)

    if ae_index == ab_index:
        print("equal ae_index", ae_index, "ae_hd", ae_hd)
        print("equal ab_index", ab_index, "ab_hd", ab_hd)

    # 将横纵坐标之和的值作为排序标准进行排序，然后进行查找，基于原数组的位置作为密钥值
    a_list.append(np.where(np.array(oneDimCSIa1) == np.array(sumEachDim(CSIa1Back, i)))[0][0])
    b_list.append(np.where(np.array(oneDimCSIb1) == np.array(sumEachDim(CSIb1Back, ab_index)))[0][0])
    e_list.append(np.where(np.array(oneDimCSIe1) == np.array(sumEachDim(CSIe1Back, ae_index)))[0][0])
    n_list.append(np.where(np.array(oneDimCSIn1) == np.array(sumEachDim(CSIn1Back, an_index)))[0][0])
    # print("\n")
    # print("\033[0;32;40mCSIa1", CSIa1Back[i], "\033[0m")
    # print("ab_hd", ab_hd, "\033[0;32;40mCSIb1", CSIb1Back[ab_index], "\033[0m")
    # print("ae_hd", ae_hd, "CSIe1", CSIe1Back[ae_index])
    # print("an_hd", an_hd, "CSIn1", CSIn1Back[an_index])

    # 比较各个独立计算的hd值的差异
    ab_max = max(ab_max, ab_hd)
    ae_max = max(ae_max, ae_hd)
    an_max = max(an_max, an_hd)

    # 绘图
    xa, ya = zip(*sortCSIa1Add[i])
    xb, yb = zip(*sortCSIb1Add[ab_index])
    xe, ye = zip(*sortCSIe1Add[ae_index])
    xn, yn = zip(*sortCSIn1Add[an_index])
    plt.figure()
    plt.plot(xa, ya, color="red", linewidth=2.5, label="a")
    plt.plot(xb, yb, color="blue", linewidth=2.5, label="b")
    plt.plot(xe, ye, color="black", linewidth=2.5, label="e")
    # plt.plot(xn, yn, color="yellow", linewidth=2.5, label="n") # 数量级差别太大，不方便显示
    plt.legend(loc='upper left')
    plt.savefig('./figures/not_convex/' + 'sabe' + str(i) + '.png')
    # plt.show()
    plt.close()

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

plt.close()
print("ab_max", ab_max, "ae_max", ae_max, "an_max", an_max)
print("keys of a:", a_list)
print("keys of b:", b_list)
print("keys of e:", e_list)
print("keys of n:", n_list)

for i in range(len(randomStep)):
    if not os.path.exists('./figures/' + "convex"):
        os.mkdir('./figures/' + "convex")

    # 由于随机产生step的算法不一定刚好满足step之和等于keyLen/2，故在每次复制值的时候需要判断
    if startIndex >= len(sortCSIa1Copy) or len(sortCSIa1Copy) - startIndex < 3:
        break
    sortCSIa1Split.append(sortCSIa1Copy[startIndex:startIndex + randomStep[i]])
    sortCSIb1Split.append(sortCSIb1Copy[startIndex:startIndex + randomStep[i]])
    sortCSIe1Split.append(sortCSIe1Copy[startIndex:startIndex + randomStep[i]])
    sortNoiseSplit.append(sortCSIn1Copy[startIndex:startIndex + randomStep[i]])
    startIndex = startIndex + randomStep[i]

sortCSIa1 = sortCSIa1Split
sortCSIb1 = sortCSIb1Split
sortCSIe1 = sortCSIe1Split
sortNoise = sortNoiseSplit

CSIa1Back = sortCSIa1
CSIb1Back = sortCSIb1
CSIe1Back = sortCSIe1
CSIn1Back = sortNoise

# for i in range(len(CSIa1Back)):
#     is_in = Polygon(CSIa1Back[i]).intersects(Polygon(sortCSIb1[i]))
#     while is_in is True:
#         for j in range(len(CSIa1Back[i])):
#             CSIa1Back[i][j] = [random.uniform(0, np.log10(np.abs(_max - _min))),
#                                random.uniform(0, np.log10(np.abs(_max - _min)))]
#         is_in = Polygon(CSIa1Back[i]).intersects(Polygon(sortCSIb1[i]))

# 凸包检查
for i in range(len(CSIa1Back)):
    CSIa1Back[i] = list(Polygon(CSIa1Back[i]).convex_hull.exterior.coords)[0:-1]
    CSIb1Back[i] = list(Polygon(CSIb1Back[i]).convex_hull.exterior.coords)[0:-1]
    CSIe1Back[i] = list(Polygon(CSIe1Back[i]).convex_hull.exterior.coords)[0:-1]
    CSIn1Back[i] = list(Polygon(CSIn1Back[i]).convex_hull.exterior.coords)[0:-1]

# 降维以用于后续的排序
oneDimCSIa1 = toOneDim(CSIa1Back)
oneDimCSIb1 = toOneDim(CSIb1Back)
oneDimCSIe1 = toOneDim(CSIe1Back)
oneDimCSIn1 = toOneDim(CSIn1Back)

# # 初始化一维数组，作为基准数组
# sortCSIa1Back = np.sort(oneDimCSIa1, axis=0)
# sortCSIb1Back = np.sort(oneDimCSIb1, axis=0)
# sortCSIe1Back = np.sort(oneDimCSIe1, axis=0)
# sortCSIn1Back = np.sort(oneDimCSIn1, axis=0)

# 计算hd距离和多边形的顺序无关，可以任意洗牌
# CSIa1Back = [[] for _ in range(len(sortCSIa1))]
# CSIb1Back = [[] for _ in range(len(sortCSIb1))]
# CSIe1Back = [[] for _ in range(len(sortCSIe1))]
# CSIn1Back = [[] for _ in range(len(sortNoise))]
#
# rand_out_polygon = list(range(len(sortCSIa1)))
# rand_in_polygon = list(range(step))
#
# random.shuffle(rand_out_polygon)
# random.shuffle(rand_in_polygon)
# for i in range(len(sortCSIa1)):
#     for j in range(step):
#         CSIa1Back[i].append(sortCSIa1[rand_out_polygon[i]][rand_in_polygon[j]])
#
# random.shuffle(rand_out_polygon)
# random.shuffle(rand_in_polygon)
# for i in range(len(sortCSIb1)):
#     for j in range(step):
#         CSIb1Back[i].append(sortCSIb1[rand_out_polygon[i]][rand_in_polygon[j]])
#
# random.shuffle(rand_out_polygon)
# random.shuffle(rand_in_polygon)
# for i in range(len(sortCSIe1)):
#     for j in range(step):
#         CSIe1Back[i].append(sortCSIe1[rand_out_polygon[i]][rand_in_polygon[j]])
#
# random.shuffle(rand_out_polygon)
# random.shuffle(rand_in_polygon)
# for i in range(len(sortNoise)):
#     for j in range(step):
#         CSIn1Back[i].append(sortNoise[rand_out_polygon[i]][rand_in_polygon[j]])

# 在数组a后面加上a[0]使之成为一个首尾封闭的多边形
sortCSIa1Add = makePolygon(CSIa1Back)
sortCSIb1Add = makePolygon(CSIb1Back)
sortCSIe1Add = makePolygon(CSIe1Back)
sortCSIn1Add = makePolygon(CSIn1Back)

# 初始化各个计算出的hd值
ab_max = 0
ae_max = 0
an_max = 0

# 最后各自的密钥
a_list = []
b_list = []
e_list = []
n_list = []

for i in range(len(sortCSIa1Add)):
    xa, ya = zip(*sortCSIa1Add[i])
    xb, yb = zip(*sortCSIb1Add[i])
    xe, ye = zip(*sortCSIe1Add[i])
    xn, yn = zip(*sortCSIn1Add[i])
    plt.figure()
    plt.plot(xa, ya, color="red", linewidth=2.5, label="a" + str(i))
    plt.plot(xb, yb, color="blue", linewidth=2.5, label="b" + str(i))
    plt.plot(xe, ye, color="black", linewidth=2.5, label="e" + str(i))
    # plt.plot(xn, yn, color="yellow", linewidth=2.5, label="n") # 数量级差别太大，不方便显示
    plt.legend(loc='upper left')
    plt.savefig('./figures/convex/' + str(i) + '.png')
    plt.close()
    # plt.show()

for i in range(len(CSIa1Back)):
    ab_hd = sys.maxsize
    ae_hd = sys.maxsize
    an_hd = sys.maxsize

    ab_index = 0
    ae_index = 0
    an_index = 0
    for j in range(len(CSIa1Back)):
        # 整体计算两个集合中每个多边形的hd值，取最匹配的（hd距离最接近的两个多边形）
        ab_d = average_hd(CSIa1Back[i], CSIb1Back[j])
        ae_d = average_hd(CSIa1Back[i], CSIe1Back[j])
        an_d = average_hd(CSIa1Back[i], CSIn1Back[j])
        if ab_d < ab_hd:
            ab_hd = ab_d
            ab_index = j
        if ae_d < ae_hd:
            ae_hd = ae_d
            ae_index = j
        if an_d < an_hd:
            an_hd = an_d
            an_index = j
    if ae_index != ab_index:
        print("not equal ae_index", ae_index, "ae_hd", ae_hd)
        print("not equal ab_index", ab_index, "ab_hd", ab_hd)

    if ae_index == ab_index:
        print("equal ae_index", ae_index, "ae_hd", ae_hd)
        print("equal ab_index", ab_index, "ab_hd", ab_hd)

    # 将横纵坐标之和的值作为排序标准进行排序，然后进行查找，基于原数组的位置作为密钥值
    a_list.append(np.where(np.array(oneDimCSIa1) == np.array(sumEachDim(CSIa1Back, i)))[0][0])
    b_list.append(np.where(np.array(oneDimCSIb1) == np.array(sumEachDim(CSIb1Back, ab_index)))[0][0])
    e_list.append(np.where(np.array(oneDimCSIe1) == np.array(sumEachDim(CSIe1Back, ae_index)))[0][0])
    n_list.append(np.where(np.array(oneDimCSIn1) == np.array(sumEachDim(CSIn1Back, an_index)))[0][0])
    # print("\n")
    # print("\033[0;32;40mCSIa1", CSIa1Back[i], "\033[0m")
    # print("ab_hd", ab_hd, "\033[0;32;40mCSIb1", CSIb1Back[ab_index], "\033[0m")
    # print("ae_hd", ae_hd, "CSIe1", CSIe1Back[ae_index])
    # print("an_hd", an_hd, "CSIn1", CSIn1Back[an_index])

    # 比较各个独立计算的hd值的差异
    ab_max = max(ab_max, ab_hd)
    ae_max = max(ae_max, ae_hd)
    an_max = max(an_max, an_hd)

    # 绘图
    xa, ya = zip(*sortCSIa1Add[i])
    xb, yb = zip(*sortCSIb1Add[ab_index])
    xe, ye = zip(*sortCSIe1Add[ae_index])
    xn, yn = zip(*sortCSIn1Add[an_index])
    plt.figure()
    plt.plot(xa, ya, color="red", linewidth=2.5, label="a")
    plt.plot(xb, yb, color="blue", linewidth=2.5, label="b")
    plt.plot(xe, ye, color="black", linewidth=2.5, label="e")
    # plt.plot(xn, yn, color="yellow", linewidth=2.5, label="n") # 数量级差别太大，不方便显示
    plt.legend(loc='upper left')
    plt.savefig('./figures/convex/' + 'sabe' + str(i) + '.png')
    # plt.show()
    plt.close()

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

plt.close()
print("ab_max", ab_max, "ae_max", ae_max, "an_max", an_max)
print("keys of a:", a_list)
print("keys of b:", b_list)
print("keys of e:", e_list)
print("keys of n:", n_list)
