import math

import matplotlib.pyplot as plt
import numpy as np

import plot_util


class ACO(object):
    def __init__(self, num_city, data, num_drones):
        self.m = 50  # 蚂蚁数量
        self.alpha = 1  # 信息素重要程度因子
        self.beta = 5  # 启发函数重要因子
        self.rho = 0.1  # 信息素挥发因子
        self.Q = 1  # 常量系数
        self.num_city = num_city  # 城市规模
        self.num_drones = num_drones  # 有多少架无人机参与任务
        self.location = data  # 城市坐标
        self.Tau = np.zeros([num_city, num_city])  # 信息素矩阵
        self.Table = [[0 for _ in range(num_city)] for _ in range(self.m)]  # 生成的蚁群
        self.iter = 1
        self.dis_mat = self.compute_dis_mat(num_city, self.location)  # 计算城市之间的距离矩阵
        self.Eta = 10.0 / self.dis_mat  # 启发式函数
        self.paths = None  # 蚁群中每个个体的长度
        # 存储存储每个温度下的最终路径，画出收敛图
        self.iter_x = []
        self.iter_y = []

        # self.start_points_idx = []
        # self.greedy_init(self.dis_mat,100,num_city)

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
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        pathlens = self.compute_paths(result)
        sortindex = np.argsort(pathlens)
        index = sortindex[0]
        result = result[index]
        for i in range(len(result) - 1):
            s = result[i]
            s2 = result[i + 1]
            self.Tau[s][s2] = 1
        self.Tau[result[-1]][result[0]] = 1
        # for i in range(num_city):
        #     for j in range(num_city):
        # return result

    # 轮盘赌选择
    def rand_choose(self, p):
        x = np.random.rand()
        for i, t in enumerate(p):
            x -= t
            if x <= 0:
                break
        return i

    # 生成蚁群
    def get_ants(self, num_city):
        for i in range(self.m):
            start = np.random.randint(num_city - 1)
            self.Table[i][0] = start
            unvisit = list([x for x in range(num_city) if x != start])
            current = start
            j = 1
            while len(unvisit) != 0:
                P = []
                # 通过信息素计算城市之间的转移概率
                for v in unvisit:
                    P.append(
                        self.Tau[current][v] ** self.alpha
                        * self.Eta[current][v] ** self.beta
                    )
                P_sum = sum(P)
                P = [x / P_sum for x in P]
                # 轮盘赌选择一个一个城市
                index = self.rand_choose(P)
                current = unvisit[index]
                self.Table[i][j] = current
                unvisit.remove(current)
                j += 1

    # 关键: 修改distance matrix，适配多旅行商问题

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        # print("Location:", location)

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

        # self.start_points_idx = to_process_idx

        # print("to process indices:", to_process_idx)
        for i in to_process_idx:
            for j in to_process_idx:
                # print("processing:", i, j, dis_mat[i][j])
                dis_mat[i][j] = np.inf

        return dis_mat

    # 计算一条路径的长度
    def compute_pathlen(self, tmp_path, dis_mat):

        # print("Start!!!")

        path = tmp_path.copy()
        if path[0] not in to_process_idx:
            path.insert(0, 0)

        if path[-1] not in to_process_idx:
            path.append(0)

        a = path[0]
        b = path[-1]

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
        # 关键：此时，原点-原点的距离不再是inf，而是0
        # print("End!!!")

        return result

    # 计算一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 更新信息素
    def update_Tau(self):
        delta_tau = np.zeros([self.num_city, self.num_city])
        paths = self.compute_paths(self.Table)
        for i in range(self.m):
            for j in range(self.num_city - 1):
                a = self.Table[i][j]
                b = self.Table[i][j + 1]
                delta_tau[a][b] = delta_tau[a][b] + self.Q / paths[i]
            a = self.Table[i][0]
            b = self.Table[i][-1]
            delta_tau[a][b] = delta_tau[a][b] + self.Q / paths[i]
        self.Tau = (1 - self.rho) * self.Tau + delta_tau

    def aco(self):
        best_lenth = math.inf
        best_path = None
        early_stop_cnt = 0
        for cnt in range(epochs):
            # 生成新的蚁群
            self.get_ants(self.num_city)  # out>>self.Table
            self.paths = self.compute_paths(self.Table)
            # 取该蚁群的最优解
            tmp_lenth = min(self.paths)
            tmp_path = self.Table[self.paths.index(tmp_lenth)]
            # 可视化初始的路径
            if cnt == 0:
                init_show = self.location[tmp_path]
                init_show = np.vstack([init_show, init_show[0]])
            # 更新最优解
            if tmp_lenth < best_lenth:
                best_lenth = tmp_lenth
                best_path = tmp_path
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if early_stop_cnt == 20:  # 若连续20次没有性能提升，则早停
                break

            # 更新信息素
            self.update_Tau()

            # 保存结果
            self.iter_x.append(cnt)
            self.iter_y.append(best_lenth)
            print(f"Epoch {cnt:3}: {best_lenth:.3f}")
        return best_lenth, best_path

    def run(self):
        best_length, best_path = self.aco()
        return self.location[best_path], best_length


seed = 42
num_drones = 20
num_city = 200
epochs = 200

# 固定随机数
np.random.seed(seed)


## 初始化坐标 (第一个点是基地的起点，起点的坐标是 0,0 )
data = [[0, 0]]
for i in range(num_city - 1):
    while True:
        x = np.random.randint(-250, 250)
        y = np.random.randint(-250, 250)
        if x != 0 or y != 0:
            break
    data.append([x, y])

# print("Start from:", data[0])

data = np.array(data)

# print(data, data.shape)

# 关键：有N架无人机，则再增加N-1个`点` (坐标是起始点)，这些点之间的距离是inf
for d in range(num_drones - 1):
    data = np.vstack([data, data[0]])
    num_city += 1  # 增加欺骗城市

to_process_idx = [0]
# print("start point:", location[0])
for d in range(1, num_drones):  # 1, ... drone-1
    # print("added base point:", location[num_city - d])
    to_process_idx.append(num_city - d)

# print(data)

# print("City len assert:", num_city, data.shape[0])

# print(show_data, show_data.shape)

aco = ACO(num_city=data.shape[0], data=data.copy(), num_drones=num_drones)
Best_path, Best = aco.run()
iterations = aco.iter_x
best_record = aco.iter_y

print(f"Best Path Length: {Best:.3f}")
plot_util.plot_results(Best_path, iterations, best_record)
