import math
import matplotlib.pyplot as plt
import numpy

my_flag = 0
# my_flag == 0 计算第一个函数最大值
# my_flag == 1 计算第二个函数最大值


class fish:
    def __init__(self, div, number, visual, step, Trynumber, delta):
#div-----------------解的自变量个数。
#number----------自变量向量
#visual-------------鱼群视觉范围
#step---------------鱼移动一步的最大步长
#Trynumber----------鱼进行捕食行为时搜索周围环境更优解的次数
#delta--------------拥挤程度，[0, 1)，影响追尾和聚群
#tag----------------公告牌，记录每次循环过后，目标函数的最优值

        # 初始化
        self.div = div
        self.number = number
        self.visual = visual
        self.step = step
        self.Trynumber = Trynumber
        self.delta = delta

    def distance(self, f):
        # 计算当前鱼对象与另一个鱼对象f之间的欧几里得距离
        num = len(self.number)           # 获取当前鱼对象的自变量个数
        res = 0.0    # 初始化距离的平方和为0
        for i in range(num):
            # 计算每个维度上的差的平方并累加
            res = res + (self.number[i] - f.number[i]) * (self.number[i] - f.number[i])
        return math.sqrt(res)

    def func(self, flag):
        # 根据标志flag计算当前位置的函数值
        num = len(self.number)
        if flag == 0:
            res = 0
            for i in range(num):
                res = res + self.number[i] * self.number[i]
            return 500 - res
        else:
            res = 0
            mul = 1
            for i in range(num):
                res = res + math.fabs(self.number[i])     # 计算绝对值和
                mul = mul * math.fabs(self.number[i])     # 计算绝对值乘积
            return 500 - (res + mul)

    def prey(self):
        # 捕食操作
        pre = self.func(my_flag)
        for i in range(self.Trynumber):
            rand = numpy.random.randint(-99, 99, self.div) / 100 * self.visual # 生成随机方向的移动步长
            for j in range(self.div):
                self.number[j] = self.number[j] + rand[j]
            cur = self.func(my_flag)
            if cur > pre:
                # 如果移动后得到更好的解,捕食成功
                # print('原始分数：' + str(pre) + '新分数：' + str(cur) + '捕食成功！！')
                return cur
            else:
                # 捕食失败
                for j in range(self.div):
                    self.number[j] = self.number[j] - rand[j]  # 如果移动后未得到更好的解，则撤销移动
        # print("捕食失败！")
        return pre

    def swarm(self, fishes):
        # 聚群行为：向视觉内鱼群中心前进step
        close_swarm = find_close_swarm(fishes, self)         # 查找视觉内的鱼群
        center_f = center_fish(close_swarm)                  # 计算视觉内鱼群的中心位置
        n = len(close_swarm) - 1                             # 视觉内的鱼群数量
        if n != 0 and (center_f.func(my_flag) / n > self.delta * self.func(my_flag)):
            # print("聚群运动")
            for i in range(self.div):
                self.number[i] = self.number[i] + self.step * center_f.number[i]  # 向中心位置靠近一步
            return self.func(my_flag)
        else:
            # print("随机运动")
            return self.rand()

    def rand(self):
        for i in range(self.div):
            self.number[i] = self.number[i] + self.step * numpy.random.uniform(-1, 1, 1)



    def follow(self, fishes):
        # 追尾行为：向着视觉内鱼群中目标函数值最优的鱼前进step
        close_swarm = find_close_swarm(fishes, self)
        best_f = best_fish(close_swarm)      # 找到视觉内的最优鱼对象
        n = len(close_swarm) - 1
        if n != 0 and (best_f.func(my_flag) / n > self.delta * self.func(my_flag)):
            # 向前移动
            # print("向前移动")
            for i in range(self.div):
                self.number[i] = self.number[i] + self.step * (best_f.number[i] - self.number[i])
            return self.func(my_flag)
        else:
            # 随机运动
            # print("随机运动")
            return self.rand()


def find_close_swarm(fishes, fish_):
    # 在种群fishes中查找fish_视觉范围内的鱼
    # 输入为fishes，是一个list型变量 和一个fish对象
    # 输出为一个fish list
    res = []
    for fi in fishes:
        if fish_.distance(fi) < fish_.visual:
            res.append(fi)            # 如果 fi 在视觉内，则将其加入到 res 列表中。
    return res


def center_fish(fishes):
    # 计算当前种群的中心位置，并将其中心位置记为certer_fish以完成聚群操作
    # 输入为fishes，是一个list型变量
    # 输出为一个fish对象
    num = len(fishes)        #计算传入鱼群列表 fishes 的长度
    if num == 0 or num == 1:
        return None
    res = fish(fishes[0].div, fishes[0].number, fishes[0].visual, fishes[0].step, fishes[0].Trynumber, fishes[0].delta)
    for i in range(fishes[0].div):
        res.number[i] = 0
    for i in range(num):
        for j in range(res.div):
            res.number[j] = res.number[j] + fishes[i].number[j]
            #将鱼群中每个鱼对象的每个维度的值加到 res 对象对应维度的值上

    return res


def best_fish(fishes):
    # 计算当前种群最优个体的位置，并将其返回用于追尾操作
    num = len(fishes)
    if num == 0 or num == 1:
        return None
    index = -1      #记录最优鱼对象的索引
    max = 0         #记录最优鱼对象的目标函数值
    for i in range(num):
        if index == -1 or max < fishes[i].func(my_flag):
            index = i
            max = fishes[i].func(my_flag)
    return fishes[index]


def main():  # 主函数
    fishes = []
    div = 3  # 解的自变量个数
    fish_num = 50  # 鱼群个体数目
    gmax = 100  # 循环最大次数
    tag = 0  # 公告牌
    visual = 1
    step = 0.2
    Trynumber = 10
    delta = 0.2
    list_of_fishes = []
    # 初始化鱼群
    for i in range(fish_num):
        # 生成每个鱼对象的初始解向量 num，在10到20之间均匀分布
        num = numpy.random.uniform(10, 20, div)
        fi = fish(div, num, visual, step, Trynumber, delta)
        fishes.append(fi)
    # 创建空列表 list_of_fishes，记录每个鱼对象每次迭代后的函数值
    for i in range(fish_num):
        list_of_fishes.append([])
    for g in range(gmax):       # 更新公告牌
        for i in range(fish_num):
            if fishes[i].func(my_flag) > tag:
                tag = fishes[i].func(my_flag)
        # # 记录每次迭代的最优值
        for i in range(fish_num):
            if g >= 50:
                list_of_fishes[i].append(fishes[i].func(my_flag))
        for i in range(fish_num):      # 根据条件选择鱼的行为
            if tag == fishes[i].func(my_flag):
                fishes[i].prey()
                continue
            tmp = numpy.random.randint(0, 3, 1)
            if tmp == 0:              #聚集
                fishes[i].swarm(fishes)
            elif tmp == 1:            #追尾
                fishes[i].follow(fishes)
            else:                      #捕食
                fishes[i].prey()
    print(tag)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    for i in range(fish_num):
        #第 i 条鱼在第 50 次迭代（索引 49）时的函数值
        if math.fabs(list_of_fishes[i][49] - 500) < 20:
            plt.plot(x, list_of_fishes[i], color='orangered', marker='o', linestyle='-', label='A')
        else:
            plt.plot(x, list_of_fishes[i], color='green', marker='*', linestyle=':', label='C')

    plt.ylim(20, 50)
    plt.ylim(-500, 500)
    plt.xlabel("Loop_time")  # X轴标签
    plt.ylabel("Value")  # Y轴标签
    plt.show()


if __name__ == '__main__':
    main()