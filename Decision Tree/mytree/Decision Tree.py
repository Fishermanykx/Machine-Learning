# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:07:15 2021

@author: illusory
"""

#库引用

import copy as cp
import numpy as np
import pandas as pd
import sklearn as sk
import graphviz as viz
import matplotlib.pyplot as plt
from sklearn import datasets,tree

#数据集建立部分
class dataSet:
    def __init__(self):
        self.label = []
        self.data = []

    def createDataSet(self):
        self.label = ['no surfacing','flippers','fish']
        self.data = [[1,1,1,0,0],[1,1,0,1,1],['yes','yes','no','no','no']]
        
    def createDataSet_lenses(self):
        self.label = ['age of the patient','spectacle prescription','astigmatic','tear production rate','contact lenses']
        self.data = [[1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3],
                [1,1,1,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1,2,2,2],
                [1,2,2,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2],
                [2,1,2,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
                ['soft','no','hard','soft','no','hard','no','soft','no','hard','no','soft','no','no','no','no','no','hard','no','soft','no']]

    def createDataSet_lenses_test(self):
        self.label = ['age of the patient','spectacle prescription','astigmatic','tear production rate','contact lenses']
        self.data = [[1,3,1],
            [1,2,2],
            [1,2,1],
            [1,2,1],
            ['no','no','no']]
        
    def createDataSet_iris(self):
        iris = datasets.load_iris()
        self.label = iris.feature_names
        self.label.append('flower type')
        self.data = np.transpose(iris.data).tolist() 
        self.data.append([iris.target_names[i] for i in iris.target])

def calEnt(dataSet):
    n = len(dataSet.data[0])                              
    iset = pd.value_counts(dataSet.data[-1])         
    p = iset/n                                       
    ent = (-p*np.log2(p)).sum()                      
    return ent

def mySplit(dataSet,axis,value):
    redataSet = cp.deepcopy(dataSet)
    del redataSet.label[axis]
    data = redataSet.data
    newdata = [[] for i in range(len(data))]
    for i in range(len(data[0])):
        if data[axis][i] == value:
            for j in range(len(data)):
                newdata[j].append(data[j][i])                
    del newdata[axis]
    redataSet.data = newdata
    return redataSet
        
def bestSplit(dataSet):
    baseEnt = calEnt(dataSet)                                
    bestGain = 0                                             
    axis = -1                                                
    for i in range(len(dataSet.data)-1):                     
        levels = pd.value_counts(dataSet.data[i]).index      
        ents = 0                                              
        for j in levels:                                     
            childSet = mySplit(dataSet,i,j)         
            childSet.data.insert(i,[j]*len(childSet.data[0]))
            ent = calEnt(childSet)                           
            ents += len(childSet.data[0])/len(dataSet.data[0])*ent 
        infoGain = baseEnt-ents                              
        if (infoGain > bestGain):
            bestGain = infoGain                              
            axis = i                                         
    return axis

def createTree(dataSet):
    featlist = cp.deepcopy(dataSet.label)                        
    classlist = pd.value_counts(dataSet.data[-1])            
    if classlist[0] == len(dataSet.data[0]) or len(dataSet.data) == 1:
        return classlist.index[0]                             
    axis = bestSplit(dataSet)
    bestfeat = featlist[axis]                                 
    myTree = {bestfeat:{}}                                    
    del featlist[axis]                                        
    valuelist = set(dataSet.data[axis])                       
    for value in valuelist:                                   
        mysplit = mySplit(dataSet,axis,value)
        myTree[bestfeat][value] = createTree(mysplit)
    return myTree

def createTree_pre(dataSet,train,test):
    featlist = cp.deepcopy(dataSet.label)                        
    classlist = pd.value_counts(dataSet.data[-1])            
    if classlist[0] == len(dataSet.data[0]) or len(dataSet.data) == 1:
        return classlist.index[0]                             
    axis = bestSplit(dataSet)
    bestfeat = featlist[axis]                                 
    myTree = {bestfeat:{}}
    del featlist[axis]                                        
    valuelist = set(dataSet.data[axis])    
    postTree = cp.deepcopy(myTree)
    for value in valuelist:                                   
        postTree[bestfeat][value] = classlist.index[0]
        mysplit = mySplit(dataSet,axis,value)
        myTree[bestfeat][value] = createTree_pre(mysplit,train,test)
    if acc_classify(train,test,myTree) >= acc_classify(train,test,postTree):
        return myTree
    else:
        return postTree

def classify(inputTree,labels, testVec):
    firstStr = next(iter(inputTree))                   
    secondDict = inputTree[firstStr]                   
    featIndex = labels.index(firstStr)                 
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            classLabel = classify(secondDict[key],labels,testVec) if type(secondDict[key]) == dict else secondDict[key]
    return classLabel

def acc_classify(train,test,Tree = None):
    inputTree = createTree(train) if Tree == None else Tree
    labels = train.label                                   
    tests = cp.deepcopy(test)
    result = []
    testlen = len(tests.data[0])
    for i in range(testlen):                      
        testVec = []
        for j in range(len(tests.data)-1):
            testVec.append(tests.data[j][i])                              
        classLabel = classify(inputTree,labels,testVec) 
        result.append(classLabel)                                        
    tests.data.append(result)
    wrong_num = 0
    for k in range(testlen):
        if tests.data[-1][k] != tests.data[-2][k]:
            wrong_num +=1
    acc = 1-wrong_num/testlen
    print(f'模型预测准确率为{acc}')
    return acc

#自建树的可视化
decisionNodeStyle = dict(boxstyle = "sawtooth", fc = "0.8")
leafNodeStyle = {"boxstyle": "round4", "fc": "0.8"}
arrowArgs = {"arrowstyle": "<-"}

# 画节点
def plotNode(nodeText, centerPt, parentPt, nodeStyle):
    createPlot.ax1.annotate(nodeText, xy = parentPt, xycoords = "axes fraction", xytext = centerPt
                            , textcoords = "axes fraction", va = "center", ha="center", bbox = nodeStyle, arrowprops = arrowArgs)

# 添加箭头上的标注文字
def plotMidText(centerPt, parentPt, lineText):
    xMid = (centerPt[0] + parentPt[0]) / 2.0
    yMid = (centerPt[1] + parentPt[1]) / 2.0 
    createPlot.ax1.text(xMid, yMid, lineText)    
    
# 画树
def plotTree(myTree, parentPt, parentValue):
    # 计算宽与高
    leafNum, treeDepth = getTreeSize(myTree) 
    # 在 1 * 1 的范围内画图，因此分母为 1
    # 每个叶节点之间的偏移量
    plotTree.xOff = plotTree.figSize / (plotTree.totalLeaf - 1)
    # 每一层的高度偏移量
    plotTree.yOff = plotTree.figSize / plotTree.totalDepth
    # 节点名称
    nodeName = list(myTree.keys())[0]
    # 根节点的起止点相同，可避免画线；如果是中间节点，则从当前叶节点的位置开始，
    #      然后加上本次子树的宽度的一半，则为决策节点的横向位置
    centerPt = (plotTree.x + (leafNum - 1) * plotTree.xOff / 2.0, plotTree.y)
    # 画出该决策节点
    plotNode(nodeName, centerPt, parentPt, decisionNodeStyle)
    # 标记本节点对应父节点的属性值
    plotMidText(centerPt, parentPt, parentValue)
    # 取本节点的属性值
    treeValue = myTree[nodeName]
    # 下一层各节点的高度
    plotTree.y = plotTree.y - plotTree.yOff
    # 绘制下一层
    for val in treeValue.keys():
        # 如果属性值对应的是字典，说明是子树，进行递归调用； 否则则为叶子节点
        if type(treeValue[val]) == dict:
            plotTree(treeValue[val], centerPt, str(val))
        else:
            plotNode(treeValue[val], (plotTree.x, plotTree.y), centerPt, leafNodeStyle)
            plotMidText((plotTree.x, plotTree.y), centerPt, str(val))
            # 移到下一个叶子节点
            plotTree.x = plotTree.x + plotTree.xOff
    # 递归完成后返回上一层
    plotTree.y = plotTree.y + plotTree.yOff
       
# 画出决策树
def createPlot(myTree):
    fig = plt.figure(1, facecolor = "white")
    fig.clf()
    axprops = {"xticks": [], "yticks": []}
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    # 定义画图的图形尺寸
    plotTree.figSize = 1.0 
    # 初始化树的总大小
    plotTree.totalLeaf, plotTree.totalDepth = getTreeSize(myTree)
    # 叶子节点的初始位置x 和 根节点的初始层高度y  
    plotTree.x = 0 
    plotTree.y = plotTree.figSize
    plotTree(myTree, (plotTree.figSize / 2.0, plotTree.y), "")
    plt.show()

def getTreeSize(myTree):
    nodeName = list(myTree.keys())[0]
    nodeValue = myTree[nodeName]
    leafNum = 0
    treeDepth = 0 
    leafDepth = 0
    for val in nodeValue.keys():
        if type(nodeValue[val]) == dict:
            leafNum += getTreeSize(nodeValue[val])[0]
            leafDepth = 1 + getTreeSize(nodeValue[val])[1] 
        else :
            leafNum += 1 
            leafDepth = 1 
        treeDepth = max(treeDepth, leafDepth)
    return leafNum, treeDepth

def mySplit_continuous(dataSet,axis,value):
    lessdataSet = cp.deepcopy(dataSet)
    moredataSet = cp.deepcopy(dataSet)
    del lessdataSet.label[axis]
    del moredataSet.label[axis]
    data = dataSet.data
    lessdata = [[] for i in range(len(data))]
    moredata = [[] for i in range(len(data))]
    for i in range(len(data[0])):
        if data[axis][i] <= value:
            for j in range(len(data)):
                lessdata[j].append(data[j][i])
        else:
            for j in range(len(data)):
                moredata[j].append(data[j][i])
    del lessdata[axis]
    del moredata[axis]
    lessdataSet.data = lessdata
    moredataSet.data = moredata
    return lessdataSet,moredataSet

def bestSplit_continuous(dataSet):
    baseEnt = calEnt(dataSet)                                
    bestGain = 0                                             
    axis = -1
    value = 0                                                
    for i in range(len(dataSet.data)-1):                     
        levels = pd.value_counts(dataSet.data[i]).index      
        infoGain = 0
        value1 = 0                                              
        for j in levels:                                     
            lessSet = mySplit_continuous(dataSet,i,j)[0]
            moreSet = mySplit_continuous(dataSet,i,j)[1]
            lessratio = len(lessSet.data[0])/len(dataSet.data[0])
            ent = lessratio*calEnt(lessSet) + (1-lessratio)*calEnt(moreSet)
            if (infoGain < baseEnt-ent):
                infoGain = baseEnt-ent
                value1 = j               
        if (infoGain > bestGain):
            bestGain = infoGain                              
            axis = i
            value = value1                                         
    return axis,value

def createTree_continuous(dataSet):
    featlist = cp.deepcopy(dataSet.label)                        
    classlist = pd.value_counts(dataSet.data[-1])            
    if classlist[0] == len(dataSet.data[0]) or len(dataSet.data) == 1:
        return classlist.index[0]                             
    axis = bestSplit_continuous(dataSet)[0]
    values = bestSplit_continuous(dataSet)[1]
    bestfeat = featlist[axis]                                 
    myTree = {bestfeat:{}}                                    
    del featlist[axis]                                        
    valuelist = [f'<={values}',f'>{values}']                       
    for value in valuelist:
        mysplit = mySplit_continuous(dataSet,axis,values)
        split = mysplit[0] if valuelist.index(value) == 0 else mysplit[1]
        myTree[bestfeat][value] = createTree_continuous(split)
    return myTree   

def plotdataSetTree(dataSet):
    Xtrain = np.transpose(dataSet.data[0:-1])
    label = list(set(dataSet.data[-1]))
    Ytrain = [label.index(i) for i in dataSet.data[-1]]
    dtr = tree.DecisionTreeClassifier(criterion='entropy')
    dtr = dtr.fit(Xtrain, Ytrain)
    tree.export_graphviz(dtr)
    dot_data = sk.tree.export_graphviz(dtr, out_file=None,
                                feature_names=dataSet.label[0:-1],
                                class_names=[i+' '+dataSet.label[-1] for i in label],
                                filled=True, rounded=True,
                                special_characters=True)
    graph = viz.Source(dot_data)
    graph.render(dataSet.label[-1])

fish = dataSet()
fish.createDataSet()
fishtest = cp.deepcopy(fish)
fishtest.data = [i[0:3] for i in fish.data]
lensestrain = dataSet()
lensestest = dataSet()
lensestrain.createDataSet_lenses()
lensestest.createDataSet_lenses_test()
iristrain = dataSet()
iristrain.createDataSet_iris()
iristest = cp.deepcopy(iristrain)
iristest.data = [i[10:15] for i in iristrain.data]
print(iristrain.data)

f = open("audiology.standardized.data")
audio_train=f.read().split("\n")[0:-1]
audiologytrain = dataSet()
audiologytrain.createDataSet()
num =len(audio_train[0].split(','))-1
audiologytrain.data = [[] for i in range(num)]
for i in audio_train:
    data = i.split(',')
    del data[-2]
    for j in range(num):
        feature = data[j]
        if feature == 't':
            feature = 1
        elif feature == 'f':
            feature = 0
        elif feature == 'normal':
            feature = 0
        elif feature == 'absent':
            feature = 1
        elif feature == 'elevated':
            feature = 2
        elif feature == 'a':
            feature = 0
        elif feature == 'as':
            feature = 1
        elif feature == 'b':
            feature = 2
        elif feature == 'ad':
            feature = 3
        elif feature == 'c':
            feature = 4
        elif feature == 'good':
            feature = 1
        elif feature == 'very_good':
            feature = 2
        elif feature == 'very_poor':
            feature = 3
        elif feature == 'poor':
            feature = 4
        elif feature == 'unmeasured':
            feature = 5
        elif feature == 'degraded':
            feature = 1
        elif feature == 'moderate':
            feature = 2
        elif feature == 'mild':
            feature = 1
        elif feature == 'severe':
            feature = 3
        elif feature == 'profound':
            feature = 4
        elif feature == '?':
            feature = 10
        audiologytrain.data[j].append(feature)
f.close()

f1 = open("audiology.standardized.test")
audio_test=f1.read().split("\n")[0:-1]
audiologytest = dataSet()
audiologytest.createDataSet()
num =len(audio_test[0].split(','))-1
audiologytest.data = [[] for i in range(num)]
for i in audio_test:
    data = i.split(',')
    del data[-2]
    for j in range(num):
        feature = data[j]
        if feature == 't':
            feature = 1
        elif feature == 'f':
            feature = 0
        elif feature == 'normal':
            feature = 0
        elif feature == 'absent':
            feature = 1
        elif feature == 'elevated':
            feature = 2
        elif feature == 'a':
            feature = 0
        elif feature == 'as':
            feature = 1
        elif feature == 'b':
            feature = 2
        elif feature == 'ad':
            feature = 3
        elif feature == 'c':
            feature = 4
        elif feature == 'good':
            feature = 1
        elif feature == 'very_good':
            feature = 2
        elif feature == 'very_poor':
            feature = 3
        elif feature == 'poor':
            feature = 4
        elif feature == 'unmeasured':
            feature = 5
        elif feature == 'degraded':
            feature = 1
        elif feature == 'moderate':
            feature = 2
        elif feature == 'mild':
            feature = 1
        elif feature == 'severe':
            feature = 3
        elif feature == 'profound':
            feature = 4            
        audiologytest.data[j].append(feature)
f1.close()

labels = ['age_gt_60','air()','airBoneGap','ar_c()','ar_u()','bone()','boneAbnormal','bser()','history_buzzing',
            'history_dizziness','history_fluctuating','history_fullness','history_heredity','history_nausea','history_noise',
            'history_recruitment','history_ringing','history_roaring','history_vomiting','late_wave_poor','m_at_2k',
            'm_cond_lt_1k','m_gt_1k','m_m_gt_2k','m_m_sn','m_m_sn_gt_1k','m_m_sn_gt_2k','m_m_sn_gt_500','m_p_sn_gt_2k','m_s_gt_500',
            'm_s_sn','m_s_sn_gt_1k','m_s_sn_gt_2k','m_s_sn_gt_3k','m_s_sn_gt_4k','m_sn_2_3k','m_sn_gt_1k','m_sn_gt_2k',
            'm_sn_gt_3k','m_sn_gt_4k','m_sn_gt_500','m_sn_gt_6k','m_sn_lt_1k','m_sn_lt_2k','m_sn_lt_3k','middle_wave_poor',
            'mod_gt_4k','mod_mixed','mod_s_mixed','mod_s_sn_gt_500','mod_sn','mod_sn_gt_1k','mod_sn_gt_2k','mod_sn_gt_3k',
            'mod_sn_gt_4k','mod_sn_gt_500','notch_4k','notch_at_4k','o_ar_c()','o_ar_u()','s_sn_gt_1k','s_sn_gt_2k','s_sn_gt_4k',
            'speech()','static_normal','tymp()','viith_nerve_signs','wave_V_delayed','waveform_ItoV_prolonged','class']

audiologytest.label = labels
audiologytrain.label = labels

audiologyTree = createTree(audiologytrain)
#createPlot(audiologyTree)

#acc_classify(audiologytrain, audiologytest)

#plotdataSetTree(iristrain)
#plotdataSetTree(lensestrain)
#plotdataSetTree(fish)
#plotdataSetTree(audiologytrain)
