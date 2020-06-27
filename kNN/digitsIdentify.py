from cv2 import *
import os
import numpy as np
import kNN


# 将待识别图片变为数字列表
def img2vector(filePath):
    img = cv2.imread(filePath)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vector = []
    for i in range(32):
        for j in range(32):
            if img[i, j] < 100:
                vector.append(1)
            else:
                vector.append(0)
    return vector


# 将文本变为数字列表
def text2vector(filePath):
    vector = np.zeros((1, 1024))
    f = open(filePath)
    for i in range(32):
        line = f.readline()
        for j in range(32):
            vector[0, 32 * i + j] = int(line[j])
    return vector


# 给出训练数据以及对应的标签
def createDataSet(file):
    trainingFileList = os.listdir(file)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))  # 训练数据
    labels = []  # 用来存放标签
    for i in range(m):
        fileName = trainingFileList[i]  # 0_1.txt, ..., 9_99.txt
        fileStr = fileName.split('.')[0]  # 0_1, ..., 9_99
        classNum = int(fileStr.split('_')[0])  # 0, 0, ..., 99, 99
        labels.append(classNum)  # 添加标签
        trainingMat[i, :] = text2vector(file + '/%s' % fileName)
    return trainingMat, labels


# kNN算法
def main():
    k = 5
    input = img2vector('4.jpg')  # 待分类数据
    trainingMat, labels = createDataSet('trainingDigits')

    reslt = kNN.classify(input, trainingMat, labels, k)
    print('The number in this image is %d.' % reslt)


if __name__ == '__main__':
    main()
