import math
import random

import matplotlib.pyplot as plt
import numpy as np

import plot_util


class PSO(object):
    def __init__(self, num_city, data):
        self.num = 200  # 粒子数目
        self.num_city = num_city  # 城市数
        self.location = data  # 城市的位置坐标
        # 计算距离矩阵
        self.dis_mat = self.compute_dis_mat(num_city, self.location)  # 计算城市之间的距离矩阵
        # 初始化所有粒子
        self.particals = self.random_init(self.num, num_city)
        # self.particals = self.greedy_init(
        #     self.dis_mat, num_total=self.num, num_city=num_city
        # )
        self.lenths = self.compute_paths(self.particals)
        # 得到初始化群体的最优解
        init_l = min(self.lenths)
        init_index = self.lenths.index(init_l)
        init_path = self.particals[init_index]
        # 画出初始的路径图
        init_show = self.location[init_path]
        # 记录每个个体的当前最优解
        self.local_best = self.particals
        self.local_best_len = self.lenths
        # 记录当前的全局最优解,长度是iteration
        self.global_best = init_path
        self.global_best_len = init_l
        # 输出解
        self.best_l = self.global_best_len
        self.best_path = self.global_best
        # 存储每次迭代的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [init_l]

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    # print("---", current, x, dis_mat[current][x])
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                if tmp_choose == -1:  # 此种情况仅可能发生在剩的都是基地点
                    tmp_choose = rest[0]
                    # print("tmp_choose:", tmp_choose)
                current = tmp_choose
                result_one.append(tmp_choose)
                # print(current, rest)
                rest.remove(tmp_choose)
                # print(rest)
            result.append(result_one)
            start_index += 1
        # print(len(result), len(result[0]))
        return result

    # 随机初始化
    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp

        for i in to_process_idx:
            for j in to_process_idx:
                # print("processing:", i, j, dis_mat[i][j])
                dis_mat[i][j] = np.inf

        return dis_mat

    # 计算路径长度
    def compute_pathlen(self, tmp_path, dis_mat):
        path = tmp_path.copy()
        if path[0] not in to_process_idx:
            path.insert(0, 0)

        if path[-1] not in to_process_idx:
            path.append(0)

        try:
            a = path[0]
            b = path[-1]
        except:
            import pdb

            pdb.set_trace()

        result = dis_mat[a][b]  # 首末城市之间的距离
        if a in to_process_idx and b in to_process_idx:
            result = 0

        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            if a in to_process_idx and b in to_process_idx:
                result += 0
            else:
                result += dis_mat[a][b]
        return result

    # 计算一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 评估当前的群体
    def eval_particals(self):
        min_lenth = min(self.lenths)
        min_index = self.lenths.index(min_lenth)
        cur_path = self.particals[min_index]
        # 更新当前的全局最优
        if min_lenth < self.global_best_len:
            self.global_best_len = min_lenth
            self.global_best = cur_path
        # 更新当前的个体最优
        for i, l in enumerate(self.lenths):
            if l < self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particals[i]

    # 粒子交叉
    def cross(self, cur, best):
        one = cur.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x = min(t)
        y = max(t)
        cross_part = best[x:y]
        tmp = []
        for t in one:
            if t in cross_part:
                continue
            tmp.append(t)
        # 两种交叉方法
        one = tmp + cross_part
        l1 = self.compute_pathlen(one, self.dis_mat)
        one2 = cross_part + tmp
        l2 = self.compute_pathlen(one2, self.dis_mat)
        if l1 < l2:
            return one, l1
        else:
            return one, l2

    # 粒子变异
    def mutate(self, one):
        one = one.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x, y = min(t), max(t)
        one[x], one[y] = one[y], one[x]
        l2 = self.compute_pathlen(one, self.dis_mat)
        return one, l2

    # 迭代操作
    def pso(self):
        early_stop_cnt = 0
        for cnt in range(epochs):
            # 更新粒子群
            for i, one in enumerate(self.particals):
                tmp_l = self.lenths[i]
                # 与当前个体局部最优解进行交叉
                new_one, new_l = self.cross(one, self.local_best[i])
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l

                # 与当前全局最优解进行交叉
                new_one, new_l = self.cross(one, self.global_best)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l
                # 变异
                one, tmp_l = self.mutate(one)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l

                # 更新该粒子
                self.particals[i] = one
                self.lenths[i] = tmp_l
            # 评估粒子群，更新个体局部最优和个体当前全局最优
            self.eval_particals()
            # 更新输出解
            if self.global_best_len < self.best_l:
                self.best_l = self.global_best_len
                self.best_path = self.global_best
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if early_stop_cnt == 50:  # 若连续50次没有性能提升，则早停
                break
            print(f"Epoch {cnt:3}: {self.best_l:.3f}")
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_l)
        return self.best_l, self.best_path

    def run(self):
        best_length, best_path = self.pso()
        # 画出最终路径
        return self.location[best_path], best_length


seed = 42
num_drones = 30
num_city = 500
epochs = 2000

# 固定随机数
np.random.seed(seed)
random.seed(seed)


## 初始化坐标 (第一个点是基地的起点，起点的坐标是 0,0 )
data = [[0, 0]]
for i in range(num_city - 1):
    while True:
        x = np.random.randint(-250, 250)
        y = np.random.randint(-250, 250)
        if x != 0 or y != 0:
            break
    data.append([x, y])

data = np.array(data)

# 关键：有N架无人机，则再增加N-1个`点` (坐标是起始点)，这些点之间的距离是inf
for d in range(num_drones - 1):
    data = np.vstack([data, data[0]])
    num_city += 1  # 增加欺骗城市

to_process_idx = [0]
for d in range(1, num_drones):  # 1, ... drone-1
    to_process_idx.append(num_city - d)

model = PSO(num_city=data.shape[0], data=data.copy())
Best_path, Best = model.run()

iterations = model.iter_x
best_record = model.iter_y

print(f"Best Path Length: {Best:.3f}")
plot_util.plot_results(Best_path, iterations, best_record)
