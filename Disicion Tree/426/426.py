import numpy as np
import pandas as pd
from numpy.core.function_base import linspace

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
def calEnt(dataSet):
    n = dataSet.shape[0]                             #数据集总行数
    iset = dataSet.iloc[:,-1].value_counts()         #标签的所有类别
    p = iset/n                                       #每一类标签所占比
    ent = (-p*np.log2(p)).sum()                      #计算信息熵
    return ent
	
    
def mySplit(dataSet,axis,value):
    '''  '''
    '''  '''
    return redataSet
    
    
def bestSplit(dataSet):
    baseEnt = calEnt(dataSet)                               #计算原始熵
    bestGain = '''  '''                                     #初始化信息增益
    axis = '''  '''                                                #初始化最佳切分列，标签列
    for i in range('''  '''):                               #对特征的每一列进行循环
        levels= '''  '''                                    #提取出当前列的所有取值
        ents = '''  '''                                             #初始化子节点的信息熵       
        for j in levels:                                     #对当前列的每一个取值进行循环
            childSet = '''  '''                              #某一个子节点的dataframe
            ent = calEnt(childSet)                           #计算某一个子节点的信息熵
            ents += '''  '''                                 #计算当前列的信息熵
        #print(f'第{i}列的信息熵为{ents}')
        infoGain = baseEnt-ents                              #计算当前列的信息增益
        #print(f'第{i}列的信息增益为{infoGain}')
        if (infoGain > bestGain):
            '''  '''                                         #选择最大信息增益
            '''  '''                                         #最大信息增益所在列的索引
    return '''  '''

    
def createTree(dataSet):
    featlist =    '''  '''                                 #提取出数据集所有的列
    classlist =    '''  '''                                #获取最后一列类标签
    #判断最多标签数目是否等于数据集行数，或者数据集是否只有一列
    if '''  ''':
        return classlist.index[0]                             #如果是，返回类标签
    axis = '''  '''                                #确定出当前最佳切分列的索引
    bestfeat = '''  '''                                 #获取该索引对应的特征
    myTree = {bestfeat:{}}                                    #采用字典嵌套的方式存储树信息
    del '''  '''                                        #删除当前特征
    valuelist = set('''  ''')                     #提取最佳切分列所有属性值
    for value in valuelist:                                   #对每一个属性值递归建树
        myTree[bestfeat][value] = createTree('''  ''')
    return myTree


def classify(inputTree,labels, testVec):
    firstStr = next(iter(inputTree))                   #获取决策树第一个节点
    secondDict = '''  '''                   #下一个字典
    featIndex = '''  '''                 #第一个节点所在列的索引
    for key in secondDict.keys():
        '''  '''
        '''  '''
        '''  '''
        '''  '''
    return '''  '''


def acc_classify(train,test):
    inputTree = '''  '''                                #根据测试集生成一棵树
    labels = '''  '''                                   #数据集所有的列名称
    result = []
    for i in range(test.shape[0]):                      #对测试集中每一条数据进行循环
        testVec = '''  '''                              #测试集中的一个实例
        classLabel = classify(inputTree,labels,testVec) #预测该实例的分类
        '''  '''                                        #将分类结果追加到result列表中
    test['predict']=result                              #将预测结果追加到测试集最后一列
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean()     #计算准确率
    print(f'模型预测准确率为{acc}')
    return test
	
	
#建立数据集
dataSet = createDataSet()
#建立决策树
myTree = createTree(dataSet)
#树的存储
np.save('myTree.npy',myTree)
#树的读取
read_myTree = np.load('myTree.npy', allow_pickle=True).item() 
#测试  
train = dataSet
test = dataSet.iloc[:3,:]
acc_classify(train,test)
#lenses数据集测试
train = createDataSet_lenses()
test = createDataSet_lenses_test()
acc_classify(train,test)