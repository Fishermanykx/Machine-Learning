import numpy as np
import pandas as pd
from numpy.core.function_base import linspace

"""
函数功能：生成数据集
参数说明：
    data：原始数据（sklearn.datasets.）
    x：数据特征
    y: 数据标签
返回：
    dataSet: 本代码使用的pd数据集
"""
def createDataSet():
    row_data = {'no surfacing':[1,1,1,0,0],
                'flippers':[1,1,0,1,1],
                'fish':['yes','yes','no','no','no']}
    dataSet = pd.DataFrame(row_data)
    return dataSet

def createDataSet_lenses():
    row_data = {'age of the patient':[1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3],
                'spectacle prescription':[1,1,1,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1,2,2,2],
                'astigmatic':[1,2,2,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2],
                'tear production rate':[2,1,2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
                'contact lenses':['soft','no','hard','soft','no','hard','no','soft','no','hard','no','soft','no','no','no','no','no','hard','no','soft','no']}
    dataSet = pd.DataFrame(row_data)
    return dataSet

def createDataSet_lenses_test():
    row_data = {'age of the patient':[1,3,1],
            'spectacle prescription':[1,2,2],
            'astigmatic':[1,2,1],
            'tear production rate':[1,2,1],
            'contact lenses':['no','no','no']}
    dataSet = pd.DataFrame(row_data)
    return dataSet

def createDataby(data,x,y):
    row_data ={}
    i=0
    for fname in data.feature_names:
        row_data[fname]=x[:, i]
        i=i+1
    targets=['' for i in range(np.size(y))]
    for i in range(np.size(y)):
        targets[i] = data.target_names[y[i]]
    row_data['target']=targets
    dataSet = pd.DataFrame(row_data)
    return dataSet


"""
函数功能：计算香农熵
参数说明：
    dataSet：原始数据集
返回：
    ent:香农熵的值
"""
def calEnt(dataSet):
    n = dataSet.shape[0]                             #数据集总行数
    iset = dataSet.iloc[:,-1].value_counts()         #标签的所有类别
    p = iset/n                                       #每一类标签所占比
    ent = (-p*np.log2(p)).sum()                      #计算信息熵
    return ent


"""
函数功能：根据信息增益选择出最佳数据集切分的列
参数说明：
    dataSet：原始数据集
返回：
    axis:数据集最佳切分列的索引
"""
    #选择最优的列进行切分
def bestSplit(dataSet):
    baseEnt = calEnt(dataSet)                                #计算原始熵
    bestGain = 0                                             #初始化信息增益
    axis = -1                                                #初始化最佳切分列，标签列
    for i in range(dataSet.shape[1]-1):                      #对特征的每一列进行循环
        levels= dataSet.iloc[:,i].value_counts().index       #提取出当前列的所有取值
        ents = 0                                             #初始化子节点的信息熵       
        for j in levels:                                     #对当前列的每一个取值进行循环
            childSet = dataSet[dataSet.iloc[:,i]==j]         #某一个子节点的dataframe
            ent = calEnt(childSet)                           #计算某一个子节点的信息熵
            ents += (childSet.shape[0]/dataSet.shape[0])*ent #计算当前列的信息熵
        #print(f'第{i}列的信息熵为{ents}')
        infoGain = baseEnt-ents                              #计算当前列的信息增益
        #print(f'第{i}列的信息增益为{infoGain}')
        if (infoGain > bestGain):
            bestGain = infoGain                              #选择最大信息增益
            axis = i                                         #最大信息增益所在列的索引
    return axis


"""
函数功能：按照给定的列划分数据集
参数说明：
    dataSet：原始数据集
    axis：指定的列索引
    value：指定的属性值
返回：
    redataSet：按照指定列索引和属性值切分后的数据集
"""
def mySplit(dataSet,axis,value):
    col = dataSet.columns[axis]
    redataSet = dataSet.loc[dataSet[col]==value,:].drop(col,axis=1)
    return redataSet


"""
函数功能：基于最大信息增益切分数据集，递归构建决策树
参数说明：
    dataSet：原始数据集（最后一列是标签）
返回：
    myTree：字典形式的树
"""
def createTree(dataSet):
    featlist = list(dataSet.columns)                          #提取出数据集所有的列
    classlist = dataSet.iloc[:,-1].value_counts()             #获取最后一列类标签
    #判断最多标签数目是否等于数据集行数，或者数据集是否只有一列
    if classlist[0]==dataSet.shape[0] or dataSet.shape[1] == 1:
        return classlist.index[0]                             #如果是，返回类标签
    axis = bestSplit(dataSet)                                 #确定出当前最佳切分列的索引
    bestfeat = featlist[axis]                                 #获取该索引对应的特征
    myTree = {bestfeat:{}}                                    #采用字典嵌套的方式存储树信息
    del featlist[axis]                                        #删除当前特征
    valuelist = set(dataSet.iloc[:,axis])                     #提取最佳切分列所有属性值
    for value in valuelist:                                   #对每一个属性值递归建树
        myTree[bestfeat][value] = createTree(mySplit(dataSet,axis,value))
    return myTree


"""
函数功能：对一个测试实例进行分类
参数说明：
    inputTree：已经生成的决策树
    labels：存储选择的最优特征标签
    testVec：测试数据列表，顺序对应原数据集
返回：
    classLabel：分类结果
"""
def classify(inputTree,labels, testVec):
    firstStr = next(iter(inputTree))                   #获取决策树第一个节点
    secondDict = inputTree[firstStr]                   #下一个字典
    featIndex = labels.index(firstStr)                 #第一个节点所在列的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict :
                classLabel = classify(secondDict[key], labels, testVec)
            else: 
                classLabel = secondDict[key]
    return classLabel


"""
函数功能：对测试集进行预测，并返回预测后的结果
参数说明：
    train：训练集
    test：测试集
返回：
    test：预测好分类的测试集
"""
def acc_classify(train,test):
    inputTree = createTree(train)                       #根据测试集生成一棵树
    labels = list(train.columns)                        #数据集所有的列名称
    result = []
    for i in range(test.shape[0]):                      #对测试集中每一条数据进行循环
        testVec = test.iloc[i,:-1]                      #测试集中的一个实例
        classLabel = classify(inputTree,labels,testVec) #预测该实例的分类
        result.append(classLabel)                       #将分类结果追加到result列表中
    test['predict']=result                              #将预测结果追加到测试集最后一列
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean()     #计算准确率
    print(f'模型预测准确率为{acc}')
    return test


#建立数据集
dataSet = createDataSet()
#建立决策树
myTree = createTree(dataSet)#该决策树仅针对category型特征，需扩展用于连续数值特征
#树的存储  可以保存模型，节省后续使用时的计算开销
np.save('myTree.npy',myTree)
#树的读取
read_myTree = np.load('myTree.npy', allow_pickle=True).item() 
#测试  
train = dataSet
test = dataSet.iloc[:3,:]
print(test)
acc_classify(train,test)

from sklearn import datasets, tree
#iris数据集实例
iris = datasets.load_iris()
Xtrain = iris.data
Ytrain = iris.target
train = createDataby(iris,Xtrain,Ytrain)
myTree = createTree(train)
#lenses数据集
train = createDataSet_lenses()
test = createDataSet_lenses_test()
acc_classify(train,test)
#audiology数据集
#读取.data .name .test文件并处理，建立决策树并测试

#sklearn方法
import graphviz#conda install python-graphviz
from sklearn.tree import DecisionTreeClassifier

Xtrain = dataSet.iloc[:,:-1]
Ytrain = dataSet.iloc[:,-1]
labels = Ytrain.unique().tolist()
Ytrain = Ytrain.apply(lambda x: labels.index(x))  #将本文转换为数字
#绘制树模型
dtr = DecisionTreeClassifier(criterion='entropy')
dtr = dtr.fit(Xtrain, Ytrain)
tree.export_graphviz(dtr)
dot_data = tree.export_graphviz(dtr, out_file=None,
                                feature_names=['no surfacing', 'flippers'],
                                class_names=['fish', 'not fish'],
                                filled=True, rounded=True,
                                special_characters=True)
#利用render方法生成图形
graph = graphviz.Source(dot_data)

graph.render("fishing")

iris = datasets.load_iris()
Xtrain = iris.data
Ytrain = iris.target
dtr = DecisionTreeClassifier(criterion='entropy')
dtr = dtr.fit(Xtrain, Ytrain)
tree.export_graphviz(dtr)
dot_data = tree.export_graphviz(dtr, out_file=None,
                                feature_names=['sepal length','sepal width','petal length', 'petal width'],
                                class_names=['setosa', 'versicolor','virginica'],
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")
