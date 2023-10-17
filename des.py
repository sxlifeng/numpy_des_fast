import time

import numpy as np

# Table
PC1_table = np.array([
    57, 49, 41, 33, 25, 17, 9,
    1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27,
    19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
    7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29,
    21, 13, 5, 28, 20, 12, 4
])
PC2_table = np.array([
    14, 17, 11, 24, 1, 5, 3, 28,
    15, 6, 21, 10, 23, 19, 12, 4,
    26, 8, 16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55, 30, 40,
    51, 45, 33, 48, 44, 49, 39, 56,
    34, 53, 46, 42, 50, 36, 29, 32
])
S_table = np.array([[14, 0, 4, 15, 13, 7, 1, 4, 2, 14, 15, 2, 11, 13, 8, 1, 3, 10, 10, 6, 6, 12, 12, 11, 5, 9, 9, 5, 0,
                     3, 7, 8, 4, 15, 1, 12, 14, 8, 8, 2, 13, 4, 6, 9, 2, 1, 11, 7, 15, 5, 12, 11, 9, 3, 7, 14, 3, 10,
                     10, 0, 5, 6, 0, 13, ],
                    [15, 3, 1, 13, 8, 4, 14, 7, 6, 15, 11, 2, 3, 8, 4, 14, 9, 12, 7, 0, 2, 1, 13, 10, 12, 6, 0, 9, 5,
                     11, 10, 5, 0, 13, 14, 8, 7, 10, 11, 1, 10, 3, 4, 15, 13, 4, 1, 2, 5, 11, 8, 6, 12, 7, 6, 12, 9, 0,
                     3, 5, 2, 14, 15, 9, ],
                    [10, 13, 0, 7, 9, 0, 14, 9, 6, 3, 3, 4, 15, 6, 5, 10, 1, 2, 13, 8, 12, 5, 7, 14, 11, 12, 4, 11, 2,
                     15, 8, 1, 13, 1, 6, 10, 4, 13, 9, 0, 8, 6, 15, 9, 3, 8, 0, 7, 11, 4, 1, 15, 2, 14, 12, 3, 5, 11,
                     10, 5, 14, 2, 7, 12, ],
                    [7, 13, 13, 8, 14, 11, 3, 5, 0, 6, 6, 15, 9, 0, 10, 3, 1, 4, 2, 7, 8, 2, 5, 12, 11, 1, 12, 10, 4,
                     14, 15, 9, 10, 3, 6, 15, 9, 0, 0, 6, 12, 10, 11, 1, 7, 13, 13, 8, 15, 9, 1, 4, 3, 5, 14, 11, 5, 12,
                     2, 7, 8, 2, 4, 14, ],
                    [2, 14, 12, 11, 4, 2, 1, 12, 7, 4, 10, 7, 11, 13, 6, 1, 8, 5, 5, 0, 3, 15, 15, 10, 13, 3, 0, 9, 14,
                     8, 9, 6, 4, 11, 2, 8, 1, 12, 11, 7, 10, 1, 13, 14, 7, 2, 8, 13, 15, 6, 9, 15, 12, 0, 5, 9, 6, 10,
                     3, 4, 0, 5, 14, 3, ],
                    [12, 10, 1, 15, 10, 4, 15, 2, 9, 7, 2, 12, 6, 9, 8, 5, 0, 6, 13, 1, 3, 13, 4, 14, 14, 0, 7, 11, 5,
                     3, 11, 8, 9, 4, 14, 3, 15, 2, 5, 12, 2, 9, 8, 5, 12, 15, 3, 10, 7, 11, 0, 14, 4, 1, 10, 7, 1, 6,
                     13, 0, 11, 8, 6, 13, ],
                    [4, 13, 11, 0, 2, 11, 14, 7, 15, 4, 0, 9, 8, 1, 13, 10, 3, 14, 12, 3, 9, 5, 7, 12, 5, 2, 10, 15, 6,
                     8, 1, 6, 1, 6, 4, 11, 11, 13, 13, 8, 12, 1, 3, 4, 7, 10, 14, 7, 10, 9, 15, 5, 6, 0, 8, 15, 0, 14,
                     5, 2, 9, 3, 2, 12, ],
                    [13, 1, 2, 15, 8, 13, 4, 8, 6, 10, 15, 3, 11, 7, 1, 4, 10, 12, 9, 5, 3, 6, 14, 11, 5, 0, 0, 14, 12,
                     9, 7, 2, 7, 2, 11, 1, 4, 14, 1, 7, 9, 4, 12, 10, 14, 8, 2, 13, 0, 15, 6, 12, 10, 9, 13, 0, 15, 3,
                     3, 5, 5, 6, 8, 11, ]], dtype=np.uint64)
P_table = np.array([
    16, 7, 20, 21, 29, 12, 28, 17,
    1, 15, 23, 26, 5, 18, 31, 10,
    2, 8, 24, 14, 32, 27, 3, 9,
    19, 13, 30, 6, 22, 11, 4, 25
])
E_table = np.array([
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
])
IP_table = np.array([
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
])
FP_table = np.array([
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41, 9, 49, 17, 57, 25
])


# End Table


def permutation_box(data: np.ndarray, box: list or np.ndarray, in_bits_width: int, out_bits_width: int):
    res = np.zeros(data.shape, dtype=np.uint64)
    for index, value in enumerate(box):
        tmp = np.left_shift(np.bitwise_and(data >> (in_bits_width - value), 1),
                            out_bits_width - 1 - index).astype(np.uint64)
        res |= tmp
    return res


def sbox(data):
    res = np.zeros(data.shape, data.dtype)
    for i in range(8):
        t = S_table[i][(data >> (42 - 6 * i)) & 0x3F]
        res |= (t << (28 - 4 * i)).astype(np.uint64)
    return res


def key_expend(main_key):
    rk = np.zeros([main_key.shape[0], 16], dtype=np.uint64)
    k = permutation_box(main_key, PC1_table, 64, 56)
    for i in range(16):
        if i in [0, 1, 8, 15]:
            k = (((k >> 28) & 0x7FFFFFF) << 29) | (((k >> 55) & 0x1) << 28) | ((k & 0x7FFFFFF) << 1) | ((k >> 27) & 0x1)
        else:
            k = (((k >> 28) & 0x3FFFFFF) << 30) | (((k >> 54) & 0x3) << 28) | ((k & 0x3FFFFFF) << 2) | ((k >> 26) & 0x3)
        rk[:, i:i+1] = permutation_box(k, PC2_table, 56, 48)
    return rk


def round_f(l, r, rk):
    o_l = r
    t = permutation_box(r, E_table, 32, 48)
    if t.shape[0] != rk.shape[0] and rk.ndim == 2:
        t = np.tile(t, [rk.shape[0], 1])
    t ^= rk
    t = sbox(t)
    t = permutation_box(t, P_table, 32, 32)
    o_r = l ^ t
    return o_l, o_r


def des_encrypt(data: np.ndarray, key):
    sub_key = key_expend(key)
    d = permutation_box(data, IP_table, 64, 64)
    l = (d >> 32) & 0xFFFFFFFF
    r = d & 0xFFFFFFFF
    for i in range(16):
        l, r = round_f(l, r, sub_key[:, i:i+1])
    d = (r << 32) | l
    d = permutation_box(d, FP_table, 64, 64)
    return d


def des_decrypt(data: np.ndarray, key):
    sub_key = key_expend(key)
    d = permutation_box(data, IP_table, 64, 64)
    l = (d >> 32) & 0xFFFFFFFF
    r = d & 0xFFFFFFFF
    for i in range(16)[::-1]:
        l, r = round_f(l, r, sub_key[:, i:i+1])
    d = (r << 32) | l
    d = permutation_box(d, FP_table, 64, 64)
    return d


def des(data: np.ndarray, key, en=True):
    if data.ndim == 1:
        data = np.array([data])
    if data.shape[1] != 8:
        raise Exception("输入数据必须每行8字节")
    data = np.frombuffer(data, dtype=np.dtype(np.uint64).newbyteorder('>')).reshape([-1, 1])

    if key.ndim == 1:
        key = np.array([key])
    if key.shape[1] != 8:
        raise Exception("密钥必须每行8字节")
    key = np.frombuffer(key, dtype=np.dtype(np.uint64).newbyteorder('>')).reshape([-1, 1])

    if data.shape[0] != 1 and key.shape[0] != 1 and data.shape[0] != key.shape[0]:
        raise Exception("数据和密钥的行数不一致，请检查输入")

    if en:
        res = des_encrypt(data, key)
    else:
        res = des_decrypt(data, key)

    res = np.frombuffer(res.astype(np.dtype(np.uint64).newbyteorder('>')), dtype=np.uint8).reshape([-1, 8])

    return res


# data = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], dtype=np.uint8)
# key = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], dtype=np.uint8)
data = np.frombuffer(int.to_bytes(0x0123456789ABCDEF, length=8, byteorder='big'), dtype=np.uint8)
cipher = np.frombuffer(int.to_bytes(0x85E813540F0AB405, length=8, byteorder='big'), dtype=np.uint8)
key = np.frombuffer(int.to_bytes(0x133457799BBCDFF1, length=8, byteorder='big'), dtype=np.uint8)

# data = np.frombuffer(int.to_bytes(0x0000000000000000, length=8, byteorder='big'), dtype=np.uint8)
# cipher = np.frombuffer(int.to_bytes(0x8CA64DE9C1B123A7, length=8, byteorder='big'), dtype=np.uint8)
# key = np.frombuffer(int.to_bytes(0x0000000000000000, length=8, byteorder='big'), dtype=np.uint8)
#
# data = np.frombuffer(int.to_bytes(0xFFFFFFFFFFFFFFFF, length=8, byteorder='big'), dtype=np.uint8)
# cipher = np.frombuffer(int.to_bytes(0x7359B2163E4EDC58, length=8, byteorder='big'), dtype=np.uint8)
# key = np.frombuffer(int.to_bytes(0xFFFFFFFFFFFFFFFF, length=8, byteorder='big'), dtype=np.uint8)
#
# data = np.frombuffer(int.to_bytes(0x0123456789ABCDEF1111111111111111, length=16, byteorder='big'), dtype=np.uint8).reshape([2, 8])
# cipher = np.frombuffer(int.to_bytes(0x8A5AE1F81AB8F2DDF40379AB9E0EC533, length=16, byteorder='big'), dtype=np.uint8).reshape([2, 8])
# key = np.frombuffer(int.to_bytes(0x1111111111111111, length=8, byteorder='big'), dtype=np.uint8)
#
key = np.frombuffer(int.to_bytes(0x0123456789ABCDEF1111111111111111, length=16, byteorder='big'), dtype=np.uint8).reshape([2, 8])
data = np.frombuffer(int.to_bytes(0x11111111111111111111111111111111, length=16, byteorder='big'), dtype=np.uint8).reshape([2, 8])
cipher = np.frombuffer(int.to_bytes(0x17668DFC7292532DF40379AB9E0EC533, length=16, byteorder='big'), dtype=np.uint8).reshape([2, 8])

print(des(data, key, True).tobytes().hex())
print(des(cipher, key, False).tobytes().hex())
print("加密校验：", (cipher == des(data, key, True)).all())
print("解密校验：", (data == des(cipher, key, False)).all())

# data = np.random.randint(1, 255, [20000, 8], dtype=np.uint8)
# key = np.frombuffer(int.to_bytes(0x1111111111111111, length=8, byteorder='big'), dtype=np.uint8)
# start = time.time()
# for i in range(50):
#     c = des(data, key, True)
# print(time.time() - start)
# print(c.shape)

