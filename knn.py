import pandas as pd
import numpy as np
import random

data = pd.read_csv(r'../test code/TWOCLASS.csv')
df = data.take(np.random.permutation(len(data)))
newDf = df.reset_index(drop=True)

dataTest = []
dataTrain = []

allData = len(newDf)
nOfTestData = len(newDf)*10/100

for i in range(0, 10):
    subDataTest = newDf[(newDf.index > -1+(i*nOfTestData))
                        & (newDf.index < (nOfTestData)+(i*nOfTestData))]
    dataTrainHead = newDf[(newDf.index > -1) & (newDf.index < i*nOfTestData)]
    dataTrainTail = newDf[(newDf.index > nOfTestData +
                           (i*nOfTestData)-1) & (newDf.index < allData + 1)]
    dataTrainAll = pd.concat([dataTrainHead, dataTrainTail])

    dataTest.append(subDataTest)
    dataTrain.append(dataTrainAll)


distanceF1 = []
distanceF2 = []
distanceF3 = []
distanceF4 = []

for i in range(0, 10):

    for j in range(0, 20):
        getDistanceF1 = []
        getDistanceF2 = []
        getDistanceF3 = []
        getDistanceF4 = []

        getClassF1 = []
        getClassF2 = []
        getClassF3 = []
        getClassF4 = []

        for k in range(0, 180):

            newIndexTrain = dataTrain[i].reset_index(drop=True)
            newIndexTest = dataTest[i].reset_index(drop=True)

            disF1 = abs(newIndexTest.f1[j] - newIndexTrain.f1[k])
            disF2 = abs(newIndexTest.f2[j] - newIndexTrain.f2[k])
            disF3 = abs(newIndexTest.f3[j] - newIndexTrain.f3[k])
            disF4 = abs(newIndexTest.f4[j] - newIndexTrain.f4[k])

            classDisF1 = newIndexTrain.classlabel[k]
            classDisF2 = newIndexTrain.classlabel[k]
            classDisF3 = newIndexTrain.classlabel[k]
            classDisF4 = newIndexTrain.classlabel[k]

            getDistanceF1.append(disF1)
            getDistanceF2.append(disF2)
            getDistanceF3.append(disF3)
            getDistanceF4.append(disF4)

            getClassF1.append(classDisF1)
            getClassF2.append(classDisF2)
            getClassF3.append(classDisF3)
            getClassF4.append(classDisF4)

        dataFrameF1 = {
            "distance": getDistanceF1,
            "clas": getClassF1}

        dataFrameF2 = {
            "distance": getDistanceF2,
            "clas": getClassF2}

        dataFrameF3 = {
            "distance": getDistanceF3,
            "clas": getClassF3}

        dataFrameF4 = {
            "distance": getDistanceF4,
            "clas": getClassF4}

    aF1 = pd.DataFrame(dataFrameF1)
    aF2 = pd.DataFrame(dataFrameF2)
    aF3 = pd.DataFrame(dataFrameF3)
    aF4 = pd.DataFrame(dataFrameF4)

    bF1 = aF1.sort_values(by=["distance"])
    bF2 = aF2.sort_values(by=["distance"])
    bF3 = aF3.sort_values(by=["distance"])
    bF4 = aF4.sort_values(by=["distance"])

    cF1 = bF1.reset_index(drop=True)
    cF2 = bF2.reset_index(drop=True)
    cF3 = bF3.reset_index(drop=True)
    cF4 = bF4.reset_index(drop=True)

    distanceF1.append(cF1)
    distanceF2.append(cF2)
    distanceF3.append(cF3)
    distanceF4.append(cF4)


# ////////////////////////////////////////////////////////////
classF1 = []
classF2 = []
classF3 = []
classF4 = []
truee = []
accuracy = []
answer = []


def knn(distance, num):
    for i in range(len(distance)):
        x = 0
        y = 0

        a = distance[i]

        for j in range(0,5):
            if a.clas[j] == 1:
                x = x+1
            elif a.clas[j] == 2:
                y = y+1

        if x > y:
            num.append(1)
        elif y > x:
            num.append(2)
        elif x == y:
            num.append(random.randrange(1, 2))


knn(distance=distanceF1, num=classF1)
knn(distance=distanceF2, num=classF2)
knn(distance=distanceF3, num=classF3)
knn(distance=distanceF4, num=classF4)


for i in range(0, 10):
    x = 0
    y = 0

    a = dataTest[i].reset_index(drop=True)

    for j in range(0, 5):
        if a.classlabel[j] == 1:
            x = x+1
        elif a.classlabel[j] == 2:
            y = y+1

    if x > y:
        truee.append(1)
    elif x < y:
        truee.append(2)
    elif x == y:
        truee.append(random.randrange(1, 2))

for i in range(0, 10):
    x = 0
    if classF1[i] == truee[i]:
        x = x+25
    if classF2[i] == truee[i]:
        x = x+25
    if classF3[i] == truee[i]:
        x = x+25
    if classF4[i] == truee[i]:
        x = x+25
    accuracy.append(x)


dataFrameAll = {
    "F1": classF1,
    "F2": classF2,
    "F3": classF3,
    "F4": classF4,
    "True": truee,
    "Accuracy": accuracy,
}
print(pd.DataFrame(dataFrameAll))
print("")
print("Accuracy of program :", sum(accuracy)/len(accuracy))
