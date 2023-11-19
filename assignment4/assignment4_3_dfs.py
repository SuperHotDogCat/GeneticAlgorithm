from typing import List, Tuple
import copy
from collections import deque
N = 3

class Env:
    def __init__(self, n: int = N)->None:
        self.left: List[List[int, int]] = [[1,1] for _ in range(n)]#渡る前の岸にいる夫婦 
        self.island: List[List[int, int]] = [[0,0] for _ in range(n)]#渡る途中の島にいる夫婦 
        self.right: List[List[int, int]] = [[0,0] for _ in range(n)]#渡る岸にいる夫婦
        self.boat: int = 0 #0: 渡る前の岸, 1: 渡る途中の島, 2: 渡る岸
    def set_state(n: int, left: List[Tuple[int, int]], island: List[Tuple[int, int]], right: List[Tuple[int, int]], boat: int):
        ret = Env(n)
        ret.left = left
        ret.island = island
        ret.right = right
        ret.boat = boat
        return ret
    def __eq__(self, other_env) -> bool:
        return (other_env.left == self.left) and (other_env.island == self.island) and (other_env.right == self.right) and (other_env.boat == self.boat)
    def __repr__(self):
        return f"State(left: {self.left} island: {self.island} right: {self.right} boat: {self.boat})\n"
    def __str__(self):
        return f"State(left: {self.left} island: {self.island} right: {self.right} boat: {self.boat})\n"
    def set_state_from_index(n,src_place, dst_place, other_place, src, dst):
        if src == 0 and dst == 1:
            return Env.set_state(n,src_place,dst_place,other_place,dst)
        elif src == 1 and dst == 0:
            return Env.set_state(n,dst_place,src_place,other_place,dst)
        #0から2にいくことは今の時点ではないが、のちの課題のために追加しておく
        elif src == 0 and dst == 2:
            return Env.set_state(n,src_place,other_place,dst_place,dst)
        elif src == 2 and dst == 0:
            return Env.set_state(n,dst_place,other_place,src_place,dst)
        
        elif src == 1 and dst == 2:
            return Env.set_state(n,other_place,src_place,dst_place,dst)
        elif src == 2 and dst == 1:
            return Env.set_state(n,other_place,dst_place,src_place,dst)

    def next_states(self, visited, n: int, src: int, dst: int):
        #srcからdstに移動する時のstatesを出力する
        places = [self.left, self.island, self.right]

        src_place = copy.deepcopy(places[src]) #移動前
        dst_place = copy.deepcopy(places[dst]) #移動前
        if (src == 0 and dst == 1) or (src == 1 and dst == 0):
            other_index = 2
        elif (src == 0 and dst == 2) or (src == 2 and dst == 0):
            other_index = 1
        elif (src == 1 and dst == 2) or (src == 2 and dst == 1):
            other_index = 0
        other_place = copy.deepcopy(places[other_index]) #移動前
        return_states = []

        #まず一回移動することを考える, 左から右へ
        once_state = [] #一回移動したものを格納する配列
        for couple_idx, couple in enumerate(src_place):
            for idx, person in enumerate(couple):
                if person == 1:
                    tmp_src = copy.deepcopy(src_place)
                    tmp_dst = copy.deepcopy(dst_place)
                    tmp_src[couple_idx][idx] = 0
                    tmp_dst[couple_idx][idx] = 1
                    once_state.append(Env.set_state_from_index(n,tmp_src, tmp_dst, other_place, src, dst))
        tmp_ret_states = []
        tmp_ret_states.extend(copy.deepcopy(once_state))
        

        second_state = [] #一回移動したものを格納する配列
        for state in once_state:
            places = [state.left, state.island, state.right]
            src_place = copy.deepcopy(places[src]) #移動前
            dst_place = copy.deepcopy(places[dst]) #移動前
            if (src == 0 and dst == 1) or (src == 1 and dst == 0):
                other_index = 2
            elif (src == 0 and dst == 2) or (src == 2 and dst == 0):
                other_index = 1
            elif (src == 1 and dst == 2) or (src == 2 and dst == 1):
                other_index = 0
            other_place = copy.deepcopy(places[other_index]) #移動前
            return_states = []
            for couple_idx, couple in enumerate(src_place):
                for idx, person in enumerate(couple):
                    if person == 1:
                            tmp_src = copy.deepcopy(src_place)
                            tmp_dst = copy.deepcopy(dst_place)
                            tmp_src[couple_idx][idx] = 0
                            tmp_dst[couple_idx][idx] = 1
                            second_state.append(Env.set_state_from_index(n,tmp_src, tmp_dst, other_place, src, dst))

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


def is_valid_state(env: Env, n: int):
    #leftに対して処理を行う
    for i in range(n):
        if env.left[i] == [0, 1]: #女性のみ
            for j in range(n):
                if env.left[j][0] == 1: #別の男がいたらFalse
                    return False
    #islandに対して処理を行う
    for i in range(n):
        if env.island[i] == [0, 1]: #女性のみ
            for j in range(n):
                if env.island[j][0] == 1: #別の男がいたらFalse
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
    island = [[0,0],[0,0],[0,0]]
    right = [[0,0],[0,0],[0,0]]
    test_state = Env.set_state(3, left, island, right, 0)
    assert is_valid_state(test_state, 3) == True
    left1 = [[1,1],[0,1],[1,1]]
    island1 = [[0,0],[0,0],[0,0]]
    right1 = [[0,0],[1,0],[0,0]]
    test_state1 = Env.set_state(3, left1, island1,right1, 0)
    assert is_valid_state(test_state1, 3) == False
    left2 = [[1,1],[0,0],[1,1]]
    island2 = [[0,0],[0,0],[0,0]]
    right2 = [[0,0],[1,1],[0,0]]
    test_state2 = Env.set_state(3, left2, island2, right2, 0)
    assert is_valid_state(test_state2, 3) == True
    left3 = [[0,1],[0,0],[1,1]]
    island3 = [[1,0],[0,1],[0,0]]
    right3 = [[0,0],[1,0],[0,0]]
    test_state3 = Env.set_state(3, left3, island3, right3, 0)
    assert is_valid_state(test_state3, 3) == False

def dfs(n = N):
    state = Env(n)
    finish_left = [[0, 0] for _ in range(n)]
    finish_island = [[0, 0] for _ in range(n)]
    finish_right = [[1, 1] for _ in range(n)]
    finish_boat = 2
    finish_state = Env.set_state(n, finish_left, finish_island, finish_right, finish_boat)

    states = deque([state])
    visited = [state] #すでに訪れたことのある状態は二度は訪れないことにする
    node_counter = 1
    moves: List[int, int] = [[0,1], [1,0], [2,1], [1,2],[0,2],[2,0]]
    while states:
        node = states.pop() #DFSなのでpop, nodeはenv　class
        for src, dst in moves:
            next_states = node.next_states(visited, n, src = src, dst = dst)
            #print(next_states)
            for state in next_states:
                if state == finish_state:
                    return node_counter
            states.extend(next_states)
            visited.extend(next_states)
            node_counter += len(next_states)
    return -1

if __name__ == "__main__":
    #test_is_valid_state()
    print(dfs(1))
    print(dfs(2))
    print(dfs(3))
    print(dfs(4))
    print(dfs(5))
    print(dfs(6)) 