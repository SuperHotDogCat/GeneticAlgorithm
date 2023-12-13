"""
Wikipediaの実装例に従って実装を行う。

遺伝的アルゴリズムは一般に以下の流れで実装される。なお、下記では個体数を N, 最大世代数を G と置く。
あらかじめ N 個の個体が入る集合を二つ用意する。以下、この二つの集合を「現世代」、「次世代」と呼ぶことにする。
現世代に N 個の個体をランダムに生成する。
評価関数により、現世代の各個体の適応度をそれぞれ計算する。
ある確率で次の3つの動作のどれかを行い、その結果を次世代に保存する。
個体を一つ選択アルゴリズムで選択する
個体を二つ選択（選択方法は後述）して交叉（後述）を行う。
個体を一つ選択して突然変異（後述）を行う。
次世代の個体数が N 個になるまで上記の動作を繰り返す。
次世代の個体数が N 個になったら次世代の内容を全て現世代に移す。
3. 以降の動作を最大世代数 G 回まで繰り返し、最終的に「現世代」の中で最も適応度の高い個体を「解」として出力する。

この時交叉は突然変異する確率より非常に大きいと考える
選択確率: 94%, 交叉確率: 5%, 突然変異:1%

この問題では3種類のGAを比較するのが目的なので、同じ選択法・個体数を使用して成績を比較する
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
    parser.add_argument("--path", "-p", type = str, default="input1.txt")
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
    entity1[idx1:idx2] = copy.copy(entity2[idx1:idx2])
    entity2[idx1:idx2] = tmp
    return entity1, entity2

def exe_crossover(entities, num_entity, T):
    """
    Tは操作数
    """
    selected_entity1 = selection(entities, scores, args.num_entity, 3)
    selected_entity2 = selection(entities, scores, args.num_entity, 3)
    selected_indices = np.random.randint(0, T, 2)
    entity1, entity2 = crossover(selected_entity1, selected_entity2, min(selected_indices), max(selected_indices))
    return entity1, entity2

def mutation(entity, idx):
    """
    entity[idx]を突然変異させる
    """

    entity = copy.copy(entity)
    entity[idx] = -1 * entity[idx]
    return entity

def calc_score(entity, ways, doprint = False):
    """
    entity: score
    """
    X_s: List = np.zeros(20).tolist()
    score = 0
    for i, sign in enumerate(entity):
        #i番目の操作
        for idx in ways[i]:
            idx -= 1 #1始まりindexだった
            X_s[idx] += sign * 1
        if doprint:
            print(X_s)
        score += X_s.count(0)
    return score

def translate_entity(entity):
    for symbol in entity:
        if symbol == 1:
            print("A", end=" ")
        elif symbol == -1:
            print("B", end=" ")
    print("")

def create_first_gen(num_entity: int, T: int):
    gen = [0] * num_entity
    for idx in range(num_entity):
        gen[idx] = np.random.choice([1, -1], size = T)
    return np.array(gen)

def testcode():
    args = make_args()
    #entity生成テスト
    test_gen = np.ones(3)
    #Score計算のテスト
    print(calc_score(test_gen, ways, True))
    #翻訳のテスト
    translate_entity(test_gen)
    print(selection(entities, scores, args.num_entity, 3))
    #print(exe_crossover(entities, args.num_entity, T))

if __name__ == "__main__":
    args = make_args()
    with open(args.path, "r") as f:
       T = int(f.readline())
       ways = [0] * T
       for i in range(T):
        ways[i] = list(map(int, f.readline().split()))
    entities = create_first_gen(args.num_entity, T) #entities 生成

    scores = [0] * args.num_entity
    #この世代のscoreを計算する
    for idx, entity in enumerate(entities):
        scores[idx] = calc_score(entity, ways)
    
    print(np.max(scores))
    results = []
    for generation in tqdm(range(args.generation)):
        #generation回だけ世代を更新する
        
        #選択, 交叉, 突然変異を行い、新しい世代を作る
        new_entities = []
        while len(new_entities) < args.num_entity:
            p = np.random.rand() #probability
            if p < 0.94:
                new_entity = selection(entities, scores, args.num_entity, 3)
                new_entities.append(new_entity)
            elif 0.94 <= p < 0.99:
                new_entity1, new_entity2 = exe_crossover(entities, args.num_entity, T)
                new_entities.append(new_entity1)
                new_entities.append(new_entity2)
            else:
                selected_entity = selection(entities, scores, args.num_entity, 3)
                selected_idx = np.random.randint(0, T)
                new_entity = mutation(selected_entity, selected_idx)
                new_entities.append(new_entity)
        entities = new_entities
        scores = [0] * len(entities)
        #この世代のscoreを計算する
        for idx, entity in enumerate(entities):
            scores[idx] = calc_score(entity, ways)
        results.append([generation + 1, np.min(scores), np.max(scores), np.mean(scores)])
    results = np.array(results)
    plt.plot(results[:,0], results[:,1])
    plt.plot(results[:,0], results[:,2])
    plt.plot(results[:,0], results[:,3])
    plt.legend(["min", "max", "mean"])
    plt.savefig("darwin.png")
    print(np.max(scores))
    