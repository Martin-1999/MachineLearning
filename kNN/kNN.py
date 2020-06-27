from numpy import *


# 给出训练数据以及对应的标签
def createDataSet():
    group = array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 通过kNN进行分类  数据特征  训练集  标签   k值
def classify(input, dataSet, label, k):
    dataSize = dataSet.shape[0]  # 数据集大小

    # 计算欧式距离
    diff = tile(input, (dataSize, 1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff, axis=1)  # 行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5  # 开根号

    # 对距离进行排序
    sortedDistIndex = argsort(dist)  # rgsort()根据元素的值从大到小对元素进行排序，返回下标

    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        # 对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # 选取出现的类别次数最多的类别
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes  # 返回标签


# 测试用例
if __name__ == '__main__':
    group, labels = createDataSet()
    classes = classify([0, 0], group, labels, 3)
    print(classes)
