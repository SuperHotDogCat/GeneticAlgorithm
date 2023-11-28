from typing import List, Tuple
import copy
from collections import deque
import heapq
N = 3

"""
Envのpropertyにscoreとdepthを追加
"""

class Env:
    def __init__(self, n: int = N)->None:
        self.left: List[List[int, int]] = [[1,1] for _ in range(n)]#渡る前の岸にいる夫婦 
        self.right: List[List[int, int]] = [[0, 0] for _ in range(n)]#渡る岸にいる夫婦
        self.boat: int = 0 #0: 渡る前の岸, 1: 渡った後の岸
        self.score: float = 0 #スコア
        self.depth: int = 0
    def set_state(n: int, left: List[Tuple[int, int]], right: List[Tuple[int, int]], boat: int, depth: int):
        ret = Env(n)
        ret.left = left
        ret.right = right
        ret.boat = boat
        ret.depth = depth
        calc_score(ret, n) #calc scoreを計算
        return ret
    def __eq__(self, other_env) -> bool:
        """
        depthとscoreは同じ状態かに関係ないとする
        """
        return (other_env.left == self.left) and (other_env.right == self.right) and (other_env.boat == self.boat)
    def __repr__(self):
        return f"State(left: {self.left} right: {self.right} boat: {self.boat} score: {self.score})\n"
    def __str__(self):
        return f"State(left: {self.left} right: {self.right} boat: {self.boat} score: {self.score})\n"
    def __lt__(self, other):
        """
        heapqを使うためのコード
        other: Envを想定している
        """
        return self.score < other.score
    def next_states(self, visited, n: int):
        left = copy.deepcopy(self.left) #移動前
        right = copy.deepcopy(self.right) #移動前
        return_states = []
        if self.boat == 0:
            #まず一回移動することを考える, 左から右へ
            once_state = [] #一回移動したものを格納する配列
            for couple_idx, couple in enumerate(left):
                for idx, person in enumerate(couple):
                    if person == 1:
                        tmp_left = copy.deepcopy(left)
                        tmp_right = copy.deepcopy(right)
                        tmp_left[couple_idx][idx] = 0
                        tmp_right[couple_idx][idx] = 1
                        once_state.append(Env.set_state(n,tmp_left, tmp_right, 1, self.depth + 1))
            
            tmp_ret_states = []
            tmp_ret_states.extend(copy.deepcopy(once_state))

            second_state = [] #一回移動したものを格納する配列
            for state in once_state:
                left = copy.deepcopy(state.left) #移動前
                right = copy.deepcopy(state.right) #移動前
                for couple_idx, couple in enumerate(left):
                    for idx, person in enumerate(couple):
                        if person == 1:
                            tmp_left = copy.deepcopy(left)
                            tmp_right = copy.deepcopy(right)
                            tmp_left[couple_idx][idx] = 0
                            tmp_right[couple_idx][idx] = 1
                            second_state.append(Env.set_state(n,tmp_left, tmp_right, 1, self.depth + 1))
            tmp_ret_states.extend(second_state)
            for state in tmp_ret_states:
                if is_valid_state(state, n):
                    #移動が有効である
                    append_flag = True
                    for done in visited:
                        if done == state:
                            append_flag = False
                            break
                    if append_flag:
                        return_states.append(state)
            return return_states

        if self.boat == 1:
            #まず一回移動することを考える, 左から右へ
            once_state = [] #一回移動したものを格納する配列
            for couple_idx, couple in enumerate(right):
                for idx, person in enumerate(couple):
                    if person == 1:
                        tmp_left = copy.deepcopy(left)
                        tmp_right = copy.deepcopy(right)
                        tmp_right[couple_idx][idx] = 0
                        tmp_left[couple_idx][idx] = 1
                        once_state.append(Env.set_state(n,tmp_left, tmp_right, 0, self.depth + 1))
            
            tmp_ret_states = []
            tmp_ret_states.extend(copy.deepcopy(once_state))

            second_state = [] #一回移動したものを格納する配列
            for state in once_state:
                left = copy.deepcopy(state.left) #移動前
                right = copy.deepcopy(state.right) #移動前
                for couple_idx, couple in enumerate(right):
                    for idx, person in enumerate(couple):
                        if person == 1:
                            tmp_left = copy.deepcopy(left)
                            tmp_right = copy.deepcopy(right)
                            tmp_right[couple_idx][idx] = 0
                            tmp_left[couple_idx][idx] = 1
                            second_state.append(Env.set_state(n,tmp_left, tmp_right, 0, self.depth + 1))
            tmp_ret_states.extend(second_state)
            for state in tmp_ret_states:
                if is_valid_state(state, n):
                    #移動が有効である
                    append_flag = True
                    for done in visited:
                        if done == state:
                            append_flag = False
                            break
                    if append_flag:
                        return_states.append(state)
            return return_states

def calc_score(env: Env, n: int):
    """
    env.score(f値)を計算する
    スライドのf^*(n) = g^*(n) + h^*(n)を用いて計算する

    h^*(n)はenv.leftに残っている人数の二倍を加算することにする(行き帰りを行うことで一人ずつ運べるため)
    もしboatがright側にある場合は1加算する
    """
    score = 0
    score += env.depth #g^*(n)
    for i in range(n):
        for man in env.left[i]:
            score += 2 * man
    if env.boat == 1:
        score += 1
    env.score = score


def is_valid_state(env: Env, n: int):
    #leftに対して処理を行う
    for i in range(n):
        if env.left[i] == [0, 1]: #女性のみ
            for j in range(n):
                if env.left[j][0] == 1: #別の男がいたらFalse
                    return False
    #rightに対して処理を行う
    for i in range(n):
        if env.right[i] == [0, 1]: #女性のみ
            for j in range(n):
                if env.right[j][0] == 1: #別の男がいたらFalse
                    return False
    
    return True

def test_is_valid_state():
    #テストコード, 本番には関係ない
    left = [[1,1],[1,1],[1,1]]
    right = [[0,0],[0,0],[0,0]]
    test_state = Env.set_state(3, left, right, 0, 0)
    assert is_valid_state(test_state, 3) == True
    left1 = [[1,1],[0,1],[1,1]]
    right1 = [[0,0],[1,0],[0,0]]
    test_state1 = Env.set_state(3, left1, right1, 0, 0)
    assert is_valid_state(test_state1, 3) == False
    left2 = [[1,1],[0,0],[1,1]]
    right2 = [[0,0],[1,1],[0,0]]
    test_state2 = Env.set_state(3, left2, right2, 0, 0)
    assert is_valid_state(test_state2, 3) == True

def Astar(n = N):
    state = Env(n)
    calc_score(state, n)
    finish_left = [[0, 0] for _ in range(n)]
    finish_right = [[1, 1] for _ in range(n)]
    finish_boat = 1
    finish_state = Env.set_state(n, finish_left, finish_right, finish_boat, 1e9)
    states = []
    heapq.heappush(states, state)
    visited = [state] #すでに訪れたことのある状態は二度は訪れないことにする
    node_counter = 1
    while states:
        node = heapq.heappop(states)
        next_states = node.next_states(visited, n)
        for state in next_states:
            if state == finish_state:
                return node_counter
        states.extend(next_states)
        visited.extend(next_states)
        node_counter += len(next_states)
    return -1

if __name__ == "__main__":
    print(Astar(1))
    print(Astar(2))
    print(Astar(3))
    print(Astar(4))
    print(Astar(5))
    print(Astar(6))