# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 19:46:40 2021

@author: illusory
"""

import numpy as np
#import sklearn as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm import tqdm

class NBC():
    def __init__(self):
        self.dataset = {"data":[],"label":[]}
        self.train = {"data":[],"label":[]}
        self.test = {"data":[],"label":[]}
        self.result = []
        
    def iris_init(self):
        iris = datasets.load_iris()
        self.dataset["label"] = [iris.target_names[i] for i in iris.target]
        self.dataset["data"] = iris.data
        self.dataset["iscon"] = [1,1,1,1]
        self.train["data"],self.test["data"],self.train["label"],self.test["label"] =train_test_split(self.dataset["data"],self.dataset["label"], random_state=1, train_size=0.7,test_size=0.3)
        
    def watermelon_init(self):
        f = open("Watermelon3.0.csv",'r',encoding="utf-8")
        file = f.readlines()[1:]
        for i in file:
            data = i.split(",")[1:]
            data[-1] = data[-1][0]
            self.dataset["data"].append([float(eval(j)) for j in data[:-1]])
            self.dataset["label"].append(eval(data[-1]))
        f.close()
        self.train,self.test = self.dataset,self.dataset
        tran = np.transpose(self.dataset['data'])
        self.dataset["iscon"] = []
        for i in tran:
            if len(i) < 5*len(set(i)):
                self.dataset["iscon"].append(1)
            else:
                self.dataset["iscon"].append(0)     

    def cal_prob(self,test_data,label):
        prob = (self.train["label"].count(label)+1)/(len(self.train['label'])+len(set(self.train['label'])))
        dataset = np.transpose(self.train['data'])
        for i in range(len(test_data)):
            temp = test_data[i]
            if self.dataset["iscon"][i] == 0:
                label_sat = 0
                data_sat = 0
                for j in range(len(self.train['label'])):
                    if self.train['label'][j] == label:
                        label_sat += 1
                        if dataset[i][j] == temp:
                            data_sat += 1
                pro = (data_sat+1)/(label_sat+len(set(dataset[i])))
            else:
                label_sat = 0
                data = []
                for j in range(len(self.train['label'])):
                    if self.train['label'][j] == label:
                        data.append(dataset[i][j])
                mean = np.mean(data)
                std = np.std(data)
                pro = stats.norm.pdf(temp, mean, std)              
            prob *= pro    
        return prob
    
    def data_classify(self,test_data):
        labels = list(set(self.train["label"]))
        maxprob = 0
        for i in labels:
            prob = self.cal_prob(test_data,i)
            if prob > maxprob:
                maxprob,label = prob,i 
        return label
            
    def acc(self):
        correct = 0
        for i in range(len(self.test["data"])):
            temp = self.data_classify(self.test["data"][i])
            self.result.append(temp)
            if temp == self.test["label"][i]:
                correct += 1
        acc = correct/len(self.test["label"])
        return acc    
        
class SPODE(NBC):
    def __init__(self):
        NBC.__init__(self)
        
    def cal_info(self,xi,xj,Dc):
        setxi = list(set(xi))
        setxj = list(set(xj))
        info = 0
        for i in setxi:
            for j in setxj:
                Dij = 0
                for k in range(len(xi)):
                    if xi[k] == i and xj[k] == j:
                        Dij += 1
                a = (Dij+1)/(Dc+len(setxi)+len(setxj))
                b = (xi.count(i)+1)/(Dc+len(setxi))
                c = (xj.count(j)+1)/(Dc+len(setxj))
                info += a*np.log(a/(b+c))
        return info
    
    def super_father(self):
        labels = list(set(self.dataset['label']))
        split = [[] for e in range(len(labels))]
        for i in range(len(self.dataset['label'])):
            for j in range(len(labels)):
                if self.dataset['label'][i] == labels[j]:
                    split[j].append(self.dataset['data'][i])
                    break
        maxinf = 0
        label = 0
        for k in range(len(self.dataset["data"][0])):
            inf = 0
            for m in split:
                Dc = len(m)
                m = [list(i) for i in np.transpose(m)]
                xj = m[k]
                info = 0
                for i in range(len(m)):
                    if i != k:
                        info += self.cal_info(m[i],xj,Dc)
                inf += info*(Dc/len(self.dataset["label"]))
            if maxinf < abs(inf):
                maxinf = abs(inf)
                label = k
        return label

    def cal_prob_father(self,test_data,label):
        prob = (self.train["label"].count(label)+1)/(len(self.train['label'])+len(set(self.train['label'])))
        dataset = np.transpose(self.train['data'])
        father = test_data[self.father]
        for i in range(len(test_data)):
            temp = test_data[i]
            if self.dataset["iscon"][i] == 0:
                label_sat = 0
                data_sat = 0
                for j in range(len(self.train['label'])):
                    if self.train['label'][j] == label:
                        if self.train["data"][j][self.father] == father:
                            label_sat+= 1
                            if dataset[i][j] == temp:
                                data_sat += 1
                pro = (data_sat+1)/(label_sat+len(set(dataset[i])))
            else:
                label_sat = 0
                data = []
                for j in range(len(self.train['label'])):
                    if self.train['label'][j] == label:
                        data.append(dataset[i][j])
                mean = np.mean(data)
                std = np.std(data)
                pro = stats.norm.pdf(temp, mean, std)              
            prob *= pro    
        return prob
    
    def new_data_classify(self,test_data):
        labels = list(set(self.train["label"]))
        maxprob = 0
        for i in labels:
            prob = self.cal_prob_father(test_data,i)
            if prob > maxprob:
                maxprob,label = prob,i 
        return label
            
    def new_acc(self):
        self.father = self.super_father()
        correct = 0
        for i in range(len(self.test["data"])):
            temp = self.new_data_classify(self.test["data"][i])
            self.result.append(temp)
            if temp == self.test["label"][i]:
                correct += 1
        acc = correct/len(self.test["label"])
        return acc 

class BN():
    def __init__(self,times,datasize = 500,avg1 = 1,avg2 = 2,var1 = 0.1,var2 = 0.1):
        self.dataset = {"data":[],"label":[]}
        self.times = times
        
        x1 = np.reshape(np.random.normal(avg1, var1, datasize), (datasize, 1))
        y1 = np.array([np.array([0]) for i in range(datasize)])
        data1 = np.concatenate((x1, y1), axis=1)
        avg1 = np.mean(np.transpose(data1)[0])
        var1 = np.std(np.transpose(data1)[0])
        
        x2 = np.reshape(np.random.normal(avg2, var2, datasize), (datasize, 1))
        y2 = np.array([np.array([1]) for i in range(datasize)])
        data2 = np.concatenate((x2, y2), axis=1)
        avg2 = np.mean(np.transpose(data2)[0])
        var2 = np.std(np.transpose(data2)[0])

        self.set = np.concatenate((data1, data2), axis=0)
        np.random.shuffle(self.set)
        self.avg = [avg1,avg2]
        self.var = [var1,var2]
        for i in self.set:
            self.dataset["data"].append(i[0])
            self.dataset["label"].append(i[-1])
        
    def cal_avg(self):
        self.est_avg = [0,0]
        for i in range(self.times):
            data1 = []
            data2 = []
            for j in self.dataset['data']:
                if abs(j-self.est_avg[0])/self.var[0] < 3:#3sigma原则
                    data1.append(j)
                elif abs(j-self.est_avg[1])/self.var[1] < 3:
                    data2.append(j)
                else:
                    com_avg = np.mean(self.est_avg)
                    if j > com_avg:
                        data2.append(j)
                    else:
                        data1.append(j)
            self.est_avg = [np.mean(data1),np.mean(data2)]
        
    def cal_acc(self):
        self.cal_avg()
        avg = (np.array(self.est_avg)-np.array(self.avg))/np.array(self.avg)
        print("两类均值预测误差率分别为：",abs(avg[0]),abs(avg[1]))
       
a = NBC()
b = NBC()
a.watermelon_init()
b.iris_init()
print("西瓜3.0数据集的分类准确率为：",a.acc())
print("鸢尾花数据集的分类准确率为： ",b.acc())
c = SPODE()
d = SPODE()
c.iris_init()
d.watermelon_init()
print("鸢尾花数据集的分类准确率为： ",c.acc())
print("西瓜3.0数据集的分类准确率为：",d.acc())
e = BN(1000)
e.cal_acc()