import sys
from math import fabs

a1P = [1.8083699487370448, 1.560275578763203, 1.7936288906830342, 1.3449137367853852, 1.5360463086895126,
       1.086277467864619, 0.9772070973709036, 1.3920398367870288, 1.4022320525800276, 0.9862658822007091,
       1.6859524919092066, 1.7943526586069796, 1.0777292760409405, 1.1090273356805407, 1.742599472384759,
       1.0297917904398208, 1.1172469198389736, 1.2891737647693726, 1.7643520694839927, 1.5949019673435116,
       1.6056582830904074, 1.8135601405984354, 1.1102093958783155, 0.998550737178813, 1.353340635300279,
       1.112449988381176, 1.8210561211630707, 1.3798030447511027, 0.9673886382505859, 1.3426922507502754,
       0.9720348973407757, 1.299601329210154, 1.8028435373630498, 1.1484599742701838, 1.8398772241465335,
       1.4844889848581433, 1.4627405118236525, 1.820881223354308, 1.3158454923188465, 1.8137517540632653,
       1.781924333780855, 1.6356916624852849, 1.5380769859876886, 1.817456561967982, 1.415933107497154,
       1.7998242655559904, 0.994131218426836, 1.5897202057274096, 1.8193039654230172, 1.121736250746813,
       1.8188648197163817, 1.2405005096426007, 1.1195431788178014, 1.8256654331198205, 1.6034993706299205,
       1.1188098907967157, 1.4916169071949914, 1.1269548760742072, 1.6058018297165646, 1.8267022936136987,
       1.3937717602301345, 1.8186890370239335, 1.8140046803708243, 1.1115551380492328, 1.817456561967982,
       1.6052273583148413, 1.8424342033772492, 1.5995408590423859, 1.1247880645761983, 1.2843511640687808,
       1.1142403080060355, 1.1430785858263524, 1.2263090845823799, 1.355374763400336, 1.6066621145275748,
       1.3777700633600045, 1.4778127175004676, 1.5615136256372515, 1.8054796186679007, 1.8504861662855765,
       1.3936105448581446, 1.0719971442907668, 1.0363471266039685, 1.8241922969391338, 1.6010064813893643,
       1.8214930577827289, 1.815246976428772, 1.1243534019531334, 1.242084671833487, 1.85070876132834,
       1.1517387569923214, 1.8551770468625208, 1.6056582830904076, 1.1140006743711526, 1.3290257112759056,
       1.107053018479881, 0.9680666134035651, 1.8074687852164517, 1.5085612823829047, 1.8057514057326327,
       1.7873989365292116, 1.8349957797665315, 1.8079196007166354, 1.552742099583282, 1.1675125234391457,
       1.8148037070902783, 1.1182220001717287, 1.3315579144911165, 1.590017977785971, 1.2645488619757264,
       1.116767026823524, 1.1088594707100075, 1.0585791590253173, 1.8118667274023275, 1.1124499883811758,
       1.5589042910603992, 1.824119862857227, 1.8454862164593204, 1.8308251568559752, 1.6141881859080318,
       1.4006765182367678, 1.0262839062760998, 1.6095718678947968, 1.843130215581722, 1.844209153866593,
       1.1476363961633005, 1.8415484684091001, 1.6175537837307459]
a1 = [1.8083699487370448, 1.1234827694327372, 1.7936288906830342, 1.5609358071464081, 1.5360463086895126,
      1.7249598907368218, 0.9772070973709036, 1.6951623322345697, 1.6858331397636184, 0.9862658822007091,
      1.6859524919092066, 1.4617904344227484, 1.4605890432933608, 1.6810319565443796, 0.9838686216652116,
      1.682958803823962, 1.6867870397834757, 0.9442543607870904, 1.7643520694839927, 1.5949019673435116,
      1.6056582830904074, 1.8135601405984354, 1.1102093958783155, 1.8089996524430794, 1.8127588205378757,
      1.112449988381176, 1.8210561211630707, 1.5934277620214612, 1.5883776991194882, 1.7976151720314324,
      1.5667845337326787, 1.5652115480502147, 1.8028435373630498, 1.1484599742701838, 1.8398772241465335,
      1.853314080612347, 1.1709080297901446, 1.820881223354308, 1.5529048644206516, 1.5433990260211112,
      1.781924333780855, 1.1316841457171711, 1.8253192621218297, 1.817456561967982, 1.1164541532253758,
      1.7998242655559904, 1.5853794091840452, 1.5897202057274096, 1.8193039654230172, 1.121736250746813,
      1.8188648197163817, 1.81780905474235, 1.1195431788178014, 1.8256654331198205, 1.6034993706299205,
      1.6000226313123274, 1.8231494193580335, 1.1269548760742072, 1.6058018297165646, 1.8267022936136987,
      1.120860350458362, 1.8186890370239335, 1.8140046803708243, 1.1115551380492328, 1.817456561967982,
      1.6052273583148413, 1.6078065164896802, 1.8256654331198205, 1.1247880645761983, 1.8321907926046186,
      1.8423816849916057, 1.1430785858263524, 1.842048590162371, 1.6102284038818013, 1.6066621145275748,
      1.8223656144447888, 1.113788816201857, 1.81070426001734, 1.8054796186679007, 1.105693231118031,
      1.8093590736215428, 1.5931323195045712, 1.5972503364020154, 1.8241922969391338, 1.1234827694327372,
      1.8214930577827289, 1.815246976428772, 1.1243534019531334, 1.8303975135290993, 1.85070876132834,
      1.1517387569923214, 1.8551770468625208, 1.6056582830904076, 1.5991490801028139, 1.8063849148785152,
      1.107053018479881, 1.806927188174795, 1.8074687852164517, 1.1075053364590168, 1.8057514057326327,
      1.5831170289549863, 1.584475870856182, 1.8079196007166354, 1.1102093958783152, 1.8115093764242833,
      1.8148037070902783, 1.1182220001717287, 1.8158667944020999, 1.590017977785971, 1.5858304744038325,
      1.8047540205211976, 1.1088594707100075, 1.808279916484813, 1.8118667274023275, 1.1124499883811758,
      1.8241922969391338, 1.6076636310511074, 1.1312563338862986, 1.8308251568559752, 1.6141881859080318,
      1.6137656463542278, 1.8397934928745523, 1.1451562493754024, 1.843130215581722, 1.844209153866593,
      1.1476363961633005, 1.8415484684091001, 1.6175537837307459]
b1 = [1.813803304663092, 1.13268651460904, 1.7972213613972206, 1.5692958622643276, 1.5435714239623655,
      1.7279747928588216, 0.9694159123539815, 1.6922004289621744, 1.6828066981763221, 0.9834759341779392,
      1.6820848789266631, 1.4585874931230012, 1.4571751520997431, 1.6773635823123854, 0.9798517589161556,
      1.6794278966121188, 1.6844862921887342, 0.9457967260479999, 1.7603219406557984, 1.590916104894123,
      1.6011905326153335, 1.8134475442648212, 1.1094660499520925, 1.807444822547311, 1.8109490279235856,
      1.1090157705111308, 1.8199823954295942, 1.591361459042092, 1.5864372633913186, 1.795277377735742,
      1.5663196215248112, 1.5644293269979836, 1.803366059042167, 1.1444704211395553, 1.839184647626194,
      1.853617103459214, 1.1722136039924793, 1.8206830118056234, 1.5499836111596887, 1.5420781463356257,
      1.7797888271696942, 1.1322596895310446, 1.8240824240027835, 1.8181378817620173, 1.1174922544404358,
      1.7988807352423877, 1.584180401486382, 1.5889809420471106, 1.8187535904977168, 1.1196956811959282,
      1.8183138876334852, 1.8170803459750837, 1.1188156515495113, 1.8236480860508868, 1.6022047320331525,
      1.5995919899703626, 1.8240824240027838, 1.1271047983648077, 1.6050894618815803, 1.826420372229227,
      1.123198075031999, 1.8210328966035232, 1.816992101642381, 1.114388554274992, 1.819631662939784,
      1.6070974320195763, 1.609523265891143, 1.826938211497937, 1.1227618173540255, 1.8319976772358961,
      1.8422763193321638, 1.1444704211395553, 1.8411925625043366, 1.6105182135929785, 1.6075265061350674,
      1.824169239491709, 1.1157214284114378, 1.8122000828518063, 1.8070838129821316, 1.1076621242768452,
      1.8108595288028353, 1.594171479114912, 1.5972562829251418, 1.8231263064744223, 1.121887985103681,
      1.8211203237768236, 1.8166389448984614, 1.1253728140876187, 1.8319124127844781, 1.8511767834109374,
      1.1477793474848277, 1.8562050856837915, 1.6065246729759222, 1.6019152023678271, 1.806903195602983,
      1.1081138086461129, 1.8067225030761813, 1.8070838129821318, 1.1072099696478683, 1.8063608923293084,
      1.584180401486382, 1.5861370252307931, 1.8096046006341608, 1.1090157705111308, 1.8107700112343637,
      1.8130914921998662, 1.114833300327073, 1.8159317687081693, 1.5904702935818498, 1.5867372941334092,
      1.8058179110351111, 1.1099158630237935, 1.8089757740948198, 1.8116643576828422, 1.109915863023793,
      1.8243428184236372, 1.6075265061350674, 1.1331129206147261, 1.8311442784902705, 1.6157396886191548,
      1.615318656611479, 1.840440681901584, 1.145714224801858, 1.8440213105097858, 1.8447670228626347,
      1.1481911962420113, 1.8401060944567578, 1.6163004304425728]


# 只匹配相等的
def genAlign(base):
    align = []
    equalNum = 0
    insNum = 0
    delNum = 0
    updNum = 0
    swpNum = 0
    for i in range(len(base)):
        if base[i] == "=":
            align.append(i)
            equalNum += 1
        elif base[i] == "+":
            delNum += 1
        elif base[i] == "-":
            insNum += 1
        elif base[i] == "~":
            updNum += 1
        elif base[i] == "^":
            swpNum += 1
    print("equalNum:", equalNum, "insNum:", insNum, "delNum:", delNum, "updNum:", updNum, "swpNum:", swpNum)
    return align


# 所有的编辑操作都进行编码成密钥
def genAlign2(base):
    align = []
    equalNum = 0
    insNum = 0
    delNum = 0
    updNum = 0
    swpNum = 0
    baseRev = base[int(len(base) / 2):] + base[0:int(len(base) / 2)]  # 从中间截断，然后拼接
    for i in range(len(baseRev)):
        if baseRev[i] == "=":
            align.append('=' + str(i))  # 加上一个操作符前缀以区分不同操作
            equalNum += 1
        elif baseRev[i] == "+":
            align.append('+' + str(i))
            delNum += 1
        elif baseRev[i] == "-":
            align.append('-' + str(i))
            insNum += 1
        elif baseRev[i] == "~":
            align.append('~' + str(i))
            updNum += 1
        elif baseRev[i] == "^":
            align.append('^' + str(i))
            swpNum += 1
    print("equalNum:", equalNum, "insNum:", insNum, "delNum:", delNum, "updNum:", updNum, "swpNum:", swpNum)
    return align


# 只匹配不相等的
def genAlign3(base):
    align = []
    equalNum = 0
    insNum = 0
    delNum = 0
    updNum = 0
    swpNum = 0
    baseRev = base[int(len(base) / 2):] + base[0:int(len(base) / 2)]  # 从中间截断，然后拼接
    for i in range(len(baseRev)):
        if baseRev[i] == "=":
            equalNum += 1
        elif baseRev[i] == "+":
            align.append('+' + str(i))
            delNum += 1
        elif baseRev[i] == "-":
            align.append('-' + str(i))
            insNum += 1
        elif baseRev[i] == "~":
            align.append('~' + str(i))
            updNum += 1
        elif baseRev[i] == "^":
            align.append('^' + str(i))
            swpNum += 1
    print("equalNum:", equalNum, "insNum:", insNum, "delNum:", delNum, "updNum:", updNum, "swpNum:", swpNum)
    return align


# def genAlign(base, strB):
#     align = []
#     j = 0
#     for i in range(len(base)):
#         if base[i] == "=":
#             align.append(strB[j])
#             j += 1
#         elif base[i] == "-":
#             j += 1
#         elif base[i] == "~":
#             j += 1
#         elif base[i] == "^":
#             align.append(strB[j + 1])
#             align.append(strB[j])
#             j += 2
#     return align


def generate(base, strB):
    align = []
    j = 0
    for i in range(len(base)):
        if base[i] == "=":
            align.append(strB[j])
            j += 1
        elif base[i] == "-":
            j += 1
        elif base[i] == "^":
            align.append(strB[j + 1])
            align.append(strB[j])
            j += 2
    return align


# def align(score, strA, strB):
#     m = len(strA)
#     n = len(strB)
#     aux = [[]] * (m + 1)
#     rule = [[]] * (m + 1)
#     for i in range(len(aux)):
#         aux[i] = [0] * (n + 1)
#         rule[i] = [""] * (n + 1)
#     rule[0][0] = ""
#     for i in range(1, m + 1):
#         rule[i][0] = rule[i - 1][0] + "-"
#         aux[i][0] = aux[i - 1][0] + score["-"]
#     for i in range(1, n + 1):
#         rule[0][i] = rule[0][i - 1] + "+"
#         aux[0][i] = aux[0][i - 1] + score["+"]
#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             if strA[i - 1] == strB[j - 1]:
#                 aux[i][j] = aux[i - 1][j - 1] + score["="]
#                 rule[i][j] = rule[i - 1][j - 1] + "="
#             else:
#                 aux[i][j] = max(aux[i - 1][j] + score["-"], aux[i][j - 1] + score["+"], aux[i - 1][j - 1] + score["~"])
#                 if aux[i][j] == aux[i - 1][j - 1] + score["~"]:
#                     rule[i][j] = rule[i - 1][j - 1] + "~"
#                 elif aux[i][j] == aux[i - 1][j] + score["-"]:
#                     rule[i][j] = rule[i - 1][j] + "-"
#                 elif aux[i][j] == aux[i][j - 1] + score["+"]:
#                     rule[i][j] = rule[i][j - 1] + "+"
#     return rule[m][n]


def alignFloat(score, arrayA, arrayB, threshold):
    # threshold = max(2, second_diffAB)
    # print("阈值", threshold)

    def equal(f1, f2):
        return fabs(f1 - f2) <= threshold

    m = len(arrayA)
    n = len(arrayB)

    sortA = arrayA.copy()
    sortB = arrayB.copy()
    sortA.sort()
    sortB.sort()
    diffA = sys.maxsize
    diffB = sys.maxsize
    second_diffA = sys.maxsize
    second_diffB = sys.maxsize
    diffAB = sys.maxsize
    second_diffAB = sys.maxsize
    for i in range(m - 1):
        diff = abs(sortA[i] - sortA[i + 1])
        if second_diffA > diff:
            if diffA < diff:
                second_diffA = diff
            else:
                second_diffA = diffA
                diffA = diff
    for i in range(n - 1):
        diff = abs(sortB[i] - sortB[i + 1])
        if second_diffB > diff:
            if diffB < diff:
                second_diffB = diff
            else:
                second_diffB = diffB
                diffB = diff
    for i in range(min(m, n)):
        diff = abs(sortA[i] - sortB[i])
        if second_diffAB > diff:
            if diffAB < diff:
                second_diffAB = diff
            else:
                second_diffAB = diffAB
                diffAB = diff
    # print("min diff of A        ", diffA)
    # print("min diff of B        ", diffB)
    print("min diff of AB       ", diffAB)
    # print("second min diff of A ", second_diffA)
    # print("second min diff of B ", second_diffB)
    print("second min diff of AB", second_diffAB)

    diffA = 0
    diffB = 0
    second_diffA = 0
    second_diffB = 0
    diffAB = 0
    second_diffAB = 0
    for i in range(m - 1):
        diff = abs(sortA[i] - sortA[i + 1])
        if second_diffA < diff:
            if diffA > diff:
                second_diffA = diff
            else:
                second_diffA = diffA
                diffA = diff
    for i in range(n - 1):
        diff = abs(sortB[i] - sortB[i + 1])
        if second_diffB < diff:
            if diffB > diff:
                second_diffB = diff
            else:
                second_diffB = diffB
                diffB = diff
    for i in range(min(m, n)):
        diff = abs(sortA[i] - sortB[i])
        if second_diffAB < diff:
            if diffAB > diff:
                second_diffAB = diff
            else:
                second_diffAB = diffAB
                diffAB = diff
    # print("max diff of A        ", diffA)
    # print("max diff of B        ", diffB)
    print("max diff of AB       ", diffAB)
    # print("second max diff of A ", second_diffA)
    # print("second max diff of B ", second_diffB)
    print("second max diff of AB", second_diffAB)

    aux = [[]] * (m + 1)
    rule = [[]] * (m + 1)
    for i in range(len(aux)):
        aux[i] = [0] * (n + 1)
        rule[i] = [""] * (n + 1)
    rule[0][0] = ""
    for i in range(1, m + 1):
        rule[i][0] = rule[i - 1][0] + "-"
        aux[i][0] = aux[i - 1][0] + score["-"]
    for i in range(1, n + 1):
        rule[0][i] = rule[0][i - 1] + "+"
        aux[0][i] = aux[0][i - 1] + score["+"]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if equal(arrayA[i - 1], arrayB[j - 1]):
                aux[i][j] = aux[i - 1][j - 1] + score["="]
                rule[i][j] = rule[i - 1][j - 1] + "="
            elif i > 1 and j > 1 and equal(arrayA[i - 1], arrayB[j - 2]) and equal(arrayA[i - 2], arrayB[j - 1]):
                aux[i][j] = min(aux[i - 1][j] + score["-"], aux[i][j - 1] + score["+"],
                                aux[i - 1][j - 1] + score["~"], aux[i - 2][j - 2] + score["^"])
                if aux[i][j] == aux[i - 1][j - 1] + score["~"]:
                    rule[i][j] = rule[i - 1][j - 1] + "~"
                elif aux[i][j] == aux[i - 1][j] + score["-"]:
                    rule[i][j] = rule[i - 1][j] + "-"
                elif aux[i][j] == aux[i][j - 1] + score["+"]:
                    rule[i][j] = rule[i][j - 1] + "+"
                elif aux[i][j] == aux[i - 2][j - 2] + score["^"]:
                    rule[i][j] = rule[i - 2][j - 2] + "^"
            else:
                aux[i][j] = min(aux[i - 1][j] + score["-"], aux[i][j - 1] + score["+"], aux[i - 1][j - 1] + score["~"])
                if aux[i][j] == aux[i - 1][j - 1] + score["~"]:
                    rule[i][j] = rule[i - 1][j - 1] + "~"
                elif aux[i][j] == aux[i - 1][j] + score["-"]:
                    rule[i][j] = rule[i - 1][j] + "-"
                elif aux[i][j] == aux[i][j - 1] + score["+"]:
                    rule[i][j] = rule[i][j - 1] + "+"
    # print("score", aux[m][n])
    return rule[m][n]


def genLongestContinuous(keys):
    tmp = 1
    longest = 0
    for i in range(1, len(keys)):
        if keys[i] - keys[i - 1] == 1:
            tmp += 1
        else:
            longest = max(longest, tmp)
            tmp = 1
    return longest


def genLongestContinuous2(keys):
    tmp = 1
    longest = 0
    for i in range(1, len(keys)):
        if int(keys[i][1:]) - int(keys[i - 1][1:]) == 1:
            tmp += 1
        else:
            longest = max(longest, tmp)
            tmp = 1
    return longest


# arrayA = [60.4413907, 37.57888647, 37.95410587, 57.37781942, 41.66437795, 19.26346875, 49.65183698, 47.08151645]
# arrayA = [37.57888647, 60.4413907, 37.95410587, 41.66437795, 57.37781942, 19.26346875, 49.65183698, 47.08151645]
# arrayA = [10.323232, 60.4413907, 37.57888647, 30.233424, 32.3424232, 37.95410587, 57.37781942, 41.66437795, 19.26346875, 49.65183698, 47.08151645]
# arrayA = [60.4413907, 37.95410587, 57.37781942, 19.26346875, 47.08151645]
arrayA = [10.4413907, 27.57888647, 37.95410587, 57.37781942, 31.66437795, 19.26346875, 49.65183698, 27.08151645]
arrayB = [60.4413907, 37.57888647, 37.95410587, 57.37781942, 41.66437795, 19.26346875, 49.65183698, 47.08151645]
# print(re.search('G\w*W\w*', strA))
rule = {"=": 8, "+": 1, "-": 2, "~": 4, "^": 0}  # B复制A的，B添加一位，B删去一位，B替换成A，B交换相邻两位
# rule = {"=": 1, "+": 0, "-": 0, "~": 0, "^": 0}  # B复制A的，B添加一位，B删去一位，B替换成A，B交换相邻两位
# ruleArray1 = alignFloat(rule, arrayA, arrayB)
# print(ruleArray1)
# alignStr1 = generate(ruleArray1, arrayA)
# print(alignStr1)

# ruleStr1 = align(rule, strA, strPB)
# print(ruleStr1)
# alignStr1 = generate(ruleStr1, strA)
# print(alignStr1)

# ruleStr2 = align(rule, strB, strPA)
# print(ruleStr2)
# alignStr2 = generate(ruleStr2, strB)
# print(alignStr2)
#
# ruleStrE = align(rule, strPA, strPB)
# print(ruleStrE)
# print(generate(ruleStrE, strPA))
# print(generate(ruleStrE, strPB))
# threshold = 0.2
# ruleStr1 = alignFloat(rule, a1P, a1, threshold)
# alignStr1 = generate(ruleStr1, a1P)
# print("ruleStr1", ruleStr1)
# ruleStr2 = alignFloat(rule, a1P, b1, threshold)
# alignStr2 = generate(ruleStr2, a1P)
# print("ruleStr2", ruleStr2)
#
# for i in range(min(len(ruleStr1), len(ruleStr2))):
#     if ruleStr1[i] != ruleStr2[i]:
#         if i >= len(a1) or i >= len(a1P): continue
#         print("\033[0;35;40m", a1[i], b1[i], a1P[i], "\033[0m")
#         print(ruleStr1[i], abs(a1[i] - a1P[i]))
#         print(ruleStr2[i], abs(a1[i] - a1P[i]))
#
# sum1 = min(len(alignStr1), len(alignStr2))
# sum2 = 0
# for i in range(0, sum1):
#     sum2 += (alignStr1[i] == alignStr2[i])
# print(sum2, sum2/sum1)
