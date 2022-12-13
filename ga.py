import math
import random

import matplotlib.pyplot as plt
import numpy as np

import plot_util


class GA(object):
    def __init__(self, num_city, num_total, data):
        self.num_city = num_city
        self.num_total = num_total
        self.scores = []
        # self.iteration = iteration
        self.location = data
        self.ga_choose_ratio = 0.2
        self.mutate_ratio = 0.05
        # fruits中存每一个个体是下标的list
        self.dis_mat = self.compute_dis_mat(num_city, data)
        # self.fruits = self.greedy_init(self.dis_mat, num_total, num_city)
        self.fruits = self.random_init(num_total, num_city)
        # 显示初始化后的最佳路径
        scores = self.compute_adp(self.fruits)
        sort_index = np.argsort(-scores)
        init_best = self.fruits[sort_index[0]]
        init_best = self.location[init_best]

        # 存储每个iteration的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [1.0 / scores[sort_index[0]]]

    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        # print("Lens:", len(result), len(result[0]))
        return result

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

    # 计算种群适应度
    def compute_adp(self, fruits):
        adp = []
        for fruit in fruits:
            if isinstance(fruit, int):
                import pdb

                pdb.set_trace()
            length = self.compute_pathlen(fruit, self.dis_mat)
            adp.append(1.0 / length)
        return np.array(adp)

    def swap_part(self, list1, list2):
        index = len(list1)
        list = list1 + list2
        list = list[::-1]
        return list[:index], list[index:]

    def ga_cross(self, x, y):
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        order = list(random.sample(path_list, 2))
        order.sort()
        start, end = order

        # 找到冲突点并存下他们的下标,x中存储的是y中的下标,y中存储x与它冲突的下标
        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = y.index(sub)
            if not (index >= start and index < end):
                x_conflict_index.append(index)

        y_confict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = x.index(sub)
            if not (index >= start and index < end):
                y_confict_index.append(index)

        assert len(x_conflict_index) == len(y_confict_index)

        # 交叉
        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp

        # 解决冲突
        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            y[i], x[j] = x[j], y[i]

        assert len(set(x)) == len_ and len(set(y)) == len_
        return list(x), list(y)

    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0 : int(ga_choose_ratio * len(sort_index))]
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.fruits[index])
            parents_score.append(scores[index])
        return parents, parents_score

    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        index1, index2 = 0, 0
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    def ga_mutate(self, gene):
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        # np.random.shuffle(tmp)
        tmp = tmp[::-1]
        gene[start:end] = tmp
        return list(gene)

    def ga(self):
        # 获得优质父代
        scores = self.compute_adp(self.fruits)
        # 选择部分优秀个体作为父代候选集合
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        # 新的种群fruits
        fruits = parents.copy()
        # 生成新的种群
        while len(fruits) < self.num_total:
            # 轮盘赌方式对父代进行选择
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            # 交叉
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            # 变异
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)
            x_adp = 1.0 / self.compute_pathlen(gene_x_new, self.dis_mat)
            y_adp = 1.0 / self.compute_pathlen(gene_y_new, self.dis_mat)
            # 将适应度高的放入种群中
            if x_adp > y_adp and (not gene_x_new in fruits):
                fruits.append(gene_x_new)
            elif x_adp <= y_adp and (not gene_y_new in fruits):
                fruits.append(gene_y_new)

        self.fruits = fruits

        return tmp_best_one, tmp_best_score

    def run(self):
        BEST_LIST = None
        best_score = -math.inf
        self.best_record = []
        early_stop_cnt = 0
        for i in range(epochs):
            tmp_best_one, tmp_best_score = self.ga()
            self.iter_x.append(i)
            self.iter_y.append(1.0 / tmp_best_score)
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if early_stop_cnt == 50:  # 若连续50次没有性能提升，则早停
                break
            self.best_record.append(1.0 / best_score)
            best_length = 1.0 / best_score
            print(f"Epoch {i:3}: {best_length:.3f}")
        # print(1.0 / best_score)
        return self.location[BEST_LIST], 1.0 / best_score


seed = 42
num_drones = 30
num_city = 500
epochs = 3000

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
# print("start point:", location[0])
for d in range(1, num_drones):  # 1, ... drone-1
    # print("added base point:", location[num_city - d])
    to_process_idx.append(num_city - d)

model = GA(num_city=data.shape[0], num_total=20, data=data.copy())
Best_path, Best = model.run()
# print(Best_path)
iterations = model.iter_x
best_record = model.iter_y

# print(Best_path)

print(f"Best Path Length: {Best:.3f}")
plot_util.plot_results(Best_path, iterations, best_record)
