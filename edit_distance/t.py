import math

from scipy.special import comb, perm

# 计算排列数
# A = perm(3, 2)
# 计算组合数
# C = comb(45, 2)
# print(A, C)

i = 50
# 2个一组交换
print("swap", pow(2, i / 2))
# 插入
print("insert", comb(i * 2, i))
# 更新
print("update", comb(i, int(i / 2)))
# 删除
print("delete", comb(i, int(i / 2)))
# 全排列
print("permute", perm(i, i))
# 幂
print("key", pow(2.0, i))

keyLen = 128
print(comb(keyLen * 4, keyLen) * comb(keyLen * 3, keyLen) * comb(keyLen * 2, keyLen) * comb(keyLen * 1, keyLen))
print(perm(keyLen, keyLen))

#
# for i in range(1, 1000):
#     if math.log10(math.factorial(i)) < math.log10(pow(2.0, i) * pow(comb(2 * i, i), 3)):
#         print(i)
#         print(math.log10(math.factorial(i)))
#         print(math.log10(pow(2.0, i) * pow(comb(2 * i, i), 3)))
#     else:
#         print(math.factorial(i))
#         print(math.log10(pow(2.0, i) * pow(comb(2 * i, i), 3)))
#         break

def C2n_n(n):
    return math.factorial(2 * n) / pow(math.factorial(n), 2)

