"""
対話型進化計算を実行することとした。
好き嫌いで1か0点
左右対称な点が1組あるごとに点数を1点加算

選択法: トーナメント法(3個取り出す)
個体数: 50とする
世代数: 100
"""

import argparse 
import copy
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", "-g", type = int, default=100,)
    parser.add_argument("--path", "-p", type = str, default=None)
    parser.add_argument("--num_entity", "-n", type = int, default=50)
    args = parser.parse_args()
    return args

def selection(entities: List[List[int]], scores: List[int], num_entity: int, selection_n: int = 5):
    indices = list(range(num_entity))
    selected_indices = np.random.choice(indices, replace = False, size = selection_n) #復元抽出とする
    selected_scores = np.array(scores)[selected_indices]
    selected_idx = np.argmax(selected_scores)
    selected_idx = selected_indices[selected_idx]
    return entities[selected_idx]

def crossover(entity1, entity2, idx1, idx2):
    """
    entity: np.ndarray [1, -1, 1, ...]のように1と-1が並んだ配列
    idx: 交叉の
    """
    entity1 = copy.copy(entity1)
    entity2 = copy.copy(entity2)
    tmp = copy.copy(entity1[idx1:idx2])
    entity1[idx1:idx2] = copy.copy(entity2[idx1:idx2])
    entity2[idx1:idx2] = tmp
    return entity1, entity2

def exe_crossover(entities, num_entity, T):
    """
    Tは操作数
    """
    selected_entity_indices = np.random.randint(0, num_entity, 2)
    selected_indices = np.random.randint(0, T, 2)
    entity1, entity2 = crossover(entities[selected_entity_indices[0]], entities[selected_entity_indices[1]], min(selected_indices), max(selected_indices))
    return entity1, entity2

def mutation(entity, idx):
    """
    entity[idx]を突然変異させる
    """

    entity = copy.copy(entity)
    if entity[idx] == "0":
        entity[idx] = 1
    elif entity[idx] == "1":
        entity[idx] = 0
    return entity

def calc_score(entity, doprint = False):
    """
    to do: あとで実装する
    """
    return

"""
entity str型: 101...の並び
decoded_entity: [-32, 31]からの値を取るx, yの組み(x, y)を24個集めたもの
"""

def to6bit(number: int)->str:
    """
    数をビット表現に変換するコード
    """
    assert number < 32 and number >= -32
    if number >= 0:
        binary_representation = bin(number)[2:]  # 正の整数の場合、単純に変換
        binary_representation = '0' * (6 - len(binary_representation)) + binary_representation  # 6ビットになるようにゼロ埋め
    else:
        binary_representation = bin(2**6 + number)[2:]  # 負の整数の場合、2の補数をとる
    return binary_representation

def to6bit_test():
    """
    テストコード
    """
    for number in range(-32, 32):
        print(to6bit(number))

def toNumber(binary_representation: str)->int:
    if binary_representation[0] == '0':  # 正の整数の場合
        return int(binary_representation, 2)
    else:  # 負の整数の場合
        inverted_binary = ''.join('1' if bit == '0' else '0' for bit in binary_representation[1:])  # 1の補数を取得
        return -int(inverted_binary, 2) - 1  # 2の補数を取得して符号を付ける

def toNumber_test():
    """
    テストコード
    """
    bits = []
    for number in range(-32, 32):
        bits.append(to6bit(number))
    
    for binary_representation in bits:
        print(toNumber(binary_representation))

def toPoints(gene: str)->List[List[int]]:
    """
    genes: (2*6*23) = 276の遺伝子文字列を点の集合に変換する
    """

    idx = 0
    points_idx = 0
    points = [0] * 23
    while idx < len(gene):
        points[points_idx] = [toNumber(gene[idx:idx+6]), toNumber(gene[idx+6:idx+12])]
        idx += 2 * 6
        points_idx += 1
    return points

def toPoints_test():
    """
    テストコード
    """
    gene = "000000" * (2 * 23)
    print(toPoints(gene))
    print(len(toPoints(gene)))

def toEntity(points: List[List[int]])->str:
    gene = ""
    for point in points:
        for p in point:
            gene += to6bit(p)
    return gene

def toEntity_test():
    """
    テストコード
    """
    points = [[-1, -1] * 23]
    print(toEntity(points))
    print(len(toEntity(points)))