import numpy as np


def sign(x):
    """
    计算输入x的符号函数，如果x > 0，返回1；如果x < 0，返回-1；如果x等于0，返回0。
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def sigmoid(x):
    """
    计算输入x的Sigmoid函数值，Sigmoid函数是一个常用的激活函数，将输入映射到0和1之间。
    """
    return 1 / (1 + np.exp(-x))
