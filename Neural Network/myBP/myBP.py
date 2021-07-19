# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:37:22 2021

@author: illusory
"""
import numpy as np
import tqdm as tqdm

class standardBP():
    def __init__(self,num,rate,k=1):
        self.num = num#神经元个数
        self.rate = rate#学习率
        self.dataset = {"data":[],"label":[]}
        f = open("Watermelon3.0.csv",'r',encoding="utf-8")
        file = f.readlines()[1:]
        for i in file:
            data = i.split(",")[1:]
            data[-1] = data[-1][0]
            self.dataset["data"].append([float(eval(j)) for j in data[:-1]])
            self.dataset["label"].append(eval(data[-1]))
        self.input_num = len(self.dataset["data"][0])
        self.input_weigh = np.random.rand(self.num,self.input_num)
        self.theta = np.random.rand(self.num)
        self.hid_output = np.zeros([len(self.dataset["data"]),self.num])
        self.output_num = k
        self.output_theta = np.random.rand(k)
        self.output_weigh = np.random.rand(self.num)
    
    def sigmond(self,x):
        return 1/(1+np.exp(-x))
    
    def neu_output(self,ori_input,theta,input_weigh):
        sum_input = np.sum(ori_input*input_weigh)
        result = self.sigmond(sum_input-theta)
        return result
    
    def net_output(self,index):
        result = 0
        for i in range(self.num):
            temp = self.neu_output(self.dataset["data"][index], self.theta[i], self.input_weigh[i])
            self.hid_output[index][i] = temp
            result += temp*self.output_weigh[i]
        result = self.sigmond(result-self.output_theta[0])
        return result    
    
    def dataset_output(self):
        result = []
        for i in range(len(self.dataset["data"])):
            temp = self.net_output(i)
            result.append(int(round(temp)))
        return result

    def fit(self):
        for i in range(len(self.dataset["data"])):
            output = self.net_output(i)
            grav_out = output*(1-output)*(self.dataset["label"][i]-output)
            for j in range(self.num):
                grav_hid = self.hid_output[i][j]*(1-self.hid_output[i][j])*grav_out*self.output_weigh[j]
                self.output_weigh[j] += self.rate*grav_out*self.hid_output[i][j]
                self.output_theta[0] -= self.rate*grav_out
                for k in range(self.input_num):
                    self.input_weigh[j][k] += self.rate*grav_hid*self.dataset["data"][i][k]
                self.theta[j] -= self.rate*grav_hid

class accumulateBP(standardBP):
    def newfit(self):
        grav_out = []
        grav_hid = np.zeros([self.num,len(self.dataset["data"])])
        for i in range(len(self.dataset["data"])):
            output = self.net_output(i)
            grav_out.append(output*(1-output)*(self.dataset["label"][i]-output))
            for j in range(self.num):
                grav_hid[j][i] = self.hid_output[i][j]*(1-self.hid_output[i][j])*grav_out[i]*self.output_weigh[j]
        gravout_avg = np.sum(grav_out)/len(grav_out)
        gravhid_avg = np.array([np.sum(e)/len(e) for e in grav_hid])
        hidout_avg = np.array([np.sum(m)/len(m) for m in np.transpose(self.hid_output)])
        self.output_weigh += self.rate*gravout_avg*hidout_avg
        self.output_theta[0] -= self.rate*gravout_avg
        data = np.array([np.sum(e)/len(e) for e in np.transpose(self.dataset["data"])])
        self.input_weigh += np.array([gravhid_avg[e]*data for e in range(self.num)])
        self.theta -= self.rate*gravhid_avg

def test(num,rate,depth,step):
    print("num=%d,\trate=%.1f"%(num,rate))
    acc = [0,0]
    for i in range(0,depth,step):
        myBP_s = standardBP(num, rate)
        myBP_a = accumulateBP(num,rate)
        for j in range(i):
            myBP_s.fit()
            myBP_a.newfit()
        m = myBP_s.dataset_output()
        n = myBP_a.dataset_output()
        length = len(myBP_s.dataset["data"])
        correct_s = 0
        correct_a = 0
        for k in range(length):
            if m[k] == myBP_s.dataset["label"][k]:
                correct_s += 1
            if n[k] == myBP_a.dataset["label"][k]:
                correct_a += 1
        acc_s = correct_s/length
        acc_a = correct_a/length
        acc[0] += acc_s*step/depth
        acc[1] += acc_a*step/depth
        #print("神经网络输出结果是:\t",m)
        #print("数据集初始分类标签:\t",myBP.dataset["label"])
        #print("在神经元个数为%d,学习率为%.2f,训练次数为%d的情况下,标准BP网络和累积BP网络的准确率分别是:\n标准BP网络：%f\n累积BP网络：%f"%(num,rate,depth,acc_s,acc_a))
        print("depth=%d\tacc_s=%f\tacc_a=%f"%(i,acc_s,acc_a))
    print("avg_acc:\tacc_s=%f\tacc_a=%f"%(acc[0],acc[1]))
test(3,0.1,1000,100)