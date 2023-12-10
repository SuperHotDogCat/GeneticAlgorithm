"""
対話型進化計算を実行することとした。
好き嫌いで1か0点
左右対称な点が1組あるごとに点数を1点加算
選択90% 交叉8% 突然変異2%

選択法: トーナメント法(3個取り出す)
個体数: 50とする
世代数: 15
"""

import argparse 
import copy
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", "-g", type = int, default=15,)
    parser.add_argument("--path", "-p", type = str, default=None)
    parser.add_argument("--num_entity", "-n", type = int, default=50)
    args = parser.parse_args()
    return args

def selection(entities: List[List[int]], scores: List[int], num_entity: int, selection_n: int = 3):
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
    entity1 = entity1[:idx1] + copy.copy(entity2[idx1:idx2]) + entity1[idx2:]
    entity2 = entity2[:idx1] + tmp + entity2[idx2:]
    return entity1, entity2

def crossover_test():
    entity1 = "1" * 7
    entity2 = "0" * 7
    idx1 = 2
    idx2 = 5
    print(crossover(entity1, entity2, idx1, idx2))

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
        entity =  entity[:idx] + "1" + entity[idx+1:]
    elif entity[idx] == "1":
        entity = entity[:idx] + "0" + entity[idx+1:]
    return entity

def mutation_test():
    """
    テストコード
    """
    entity = "101111"
    print(mutation(entity, 0))
    print(mutation(entity, 3))

def calc_score(entity, doprint = False):
    """
    to do: あとで実装する
    """
    points: List[List[int]] = toPoints(entity)
    score = 0
    seen = []
    for i in range(23-1):
        for j in range(i+1, 23):
            if (points[i][0] == -points[j][0] and points[i][1] == points[j][1]) and (not points[i] in seen) and (not points[j] in seen):
                #x座標はマイナスをかけて等しくて, y座標は等しいなら左右対称
                score += 1
                seen.append(points[i])
                seen.append(points[j])
    alpha = 1 / 20 / 20
    for i in range(23-1):
        for j in range(i+1, 23):
            L2 = ((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)**(1/2)
            score -= ((20 - L2) ** 2)**(1/2) * alpha 
    return score

def climbing_score(entity):
    """
    山登り法でスコアを計算する, 1回indexを入れ替えた状態での最適値を求める
    """
    max_score = calc_score(entity)
    max_entity = entity
    T = 2 * 6 * 23
    for i in range(T-1):
        for j in range(i+1, T):
            new_entity = copy.copy(entity)
            new_entity = new_entity[:i] + entity[j] + entity[i+1:j] + entity[i] + entity[j+1:]
            new_score = calc_score(new_entity)
            if new_score > max_score:
                max_score = new_score
                max_entity = new_entity
    return max_score, max_entity


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

def create_first_gen(num_entity: int):
    gen = [0] * num_entity
    for idx in range(num_entity):
        entity = ""
        for i in range(6 * 2 * 23):
            entity += str(np.random.choice([1, 0]))
        gen[idx] = entity
    return gen

def main():
    """
    mainの実行をする
    """
    args = make_args()
    if args.path == None:
        entities = create_first_gen(args.num_entity) #entities 生成
    else:
        entities = pickle.load(open(args.path, "rb"))
    scores = [0] * args.num_entity
    #この世代のscoreを計算する
    for idx, entity in tqdm(enumerate(entities)):
        #scores[idx] = calc_score(entity, ways)
        scores[idx], entities[idx] = climbing_score(entity)
    
    results = []
    T = 2 * 6 * 23 #遺伝子長(entityの長さ)
    for generation in tqdm(range(args.generation)):
        #generation回だけ世代を更新する
        
        #選択, 交叉, 突然変異を行い、新しい世代を作る
        new_entities = []
        while len(new_entities) < args.num_entity:
            p = np.random.rand() #probability
            if p < 0.90:
                new_entity = selection(entities, scores, args.num_entity, 3)
                new_entities.append(new_entity)
            elif 0.90 <= p < 0.98:
                new_entity1, new_entity2 = exe_crossover(entities, args.num_entity, 2 * 6 * 23)
                new_entities.append(new_entity1)
                new_entities.append(new_entity2)
            else:
                selected_entity_idx = np.random.randint(0, args.num_entity)
                selected_idx = np.random.randint(0, 2 * 6 * 23)
                new_entity = mutation(entities[selected_entity_idx], selected_idx)
                new_entities.append(new_entity)
        entities = new_entities
        scores = [0] * len(entities)
        #この世代のscoreを計算する
        for idx, entity in enumerate(entities):
            #scores[idx] = calc_score(entity, ways)
            scores[idx], entities[idx] = climbing_score(entity)
        results.append([generation + 1, np.min(scores), np.max(scores), np.mean(scores)])
    results = np.array(results)
    plt.plot(results[:,0], results[:,1])
    plt.plot(results[:,0], results[:,2])
    plt.plot(results[:,0], results[:,3])
    plt.legend(["min", "max", "mean"])
    plt.savefig("points_score_lamarck.png")
    plt.close()
    print(np.max(scores))
    scores = np.array(scores)
    max_value = np.max(scores)
    max_indices = np.where(scores == max_value)[0]
    max_entities = []
    for max_idx in max_indices:
        max_entities.append(entities[max_idx])
    with open("entities_lamarck.bin", "wb") as f:
        pickle.dump(entities, f)
    return max_entities

if __name__ == "__main__":
    max_entities = main()
    print(len(max_entities))
    for idx, max_entity in enumerate(max_entities):
        points = toPoints(max_entity)
        points = np.array(points)
        plt.xlim(-32, 32)
        plt.ylim(-32, 32)
        plt.scatter(points[:,0], points[:,1])
        plt.savefig(f"points{idx}_lamarck.png")
        plt.close()