#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:22:07 2021

@author: Manting
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import argparse
import math
from sklearn.preprocessing import OneHotEncoder

#%% read data
train_data = np.load('train.npz') 
test_data = np.load('test.npz')

x_train = np.array([i.ravel() for i in (train_data['image'] / 255)]) # 51000*1024
y_train = train_data['label']

x_test = np.array([i.ravel() for i in (test_data['image'] / 255)]) #7954*1024
y_test = test_data['label']

params = json.load(open('config.json',))

#%%
def plot(train_loss, train_acc, test_loss, test_acc):
    plt.title("learning curves of J(w)")
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    
    plt.title("accuracy of classication")
    plt.plot(train_acc, label='train accuracy')
    plt.plot(test_acc, label='test accuracy')
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

#%%
class FNN():
    
     def __init__(self, batch_size, act, layer, weights = 0.1, epoch = 20,iteration = 25, learning_rate = 0.001):
        self.epoch = epoch
        self.iteration = iteration
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.act = act
        self.layer = layer
        self.weights = self.weight_init(layer, weights)
    
     def weight_init(self,nn, weights):
        layer = {}
        num = 0
        
        if weights == 0.1:
            for i in nn.values():
                w = 0.1 * np.random.randn(i['input_dim'], i['output_dim'])
                layer[num] = w
                num += 1
        elif weights == 0:
            for i in nn.values():
                w = np.zeros((i['input_dim'], i['output_dim']))
                layer[num] = w
                num += 1
        else:
            for i in weights.values():
                layer[num] = 0.1 * i
                num += 1
                
        return layer
    
     def predict(self, x): 
        predict_a, predict_z = self.forward(x)
        y = np.argmax(predict_a[-1], axis = 1)
        return y
     
     def forward(self, train):
        z = []
        a = []
        for i in range(len(self.weights)):
            print(self.weights)
            zz = 0
            if i == 0:
                print('layer0:', train.shape, self.weights[i].shape)
                zz = np.nan_to_num(np.dot(train, self.weights[i]))
                z.append(zz)
            else:
                print('layer%s'%i,np.array(a[-1]).shape, self.weights[i].shape)
                zz = np.nan_to_num(np.dot(a[-1], self.weights[i]))
                z.append(zz)
                
            if self.act[i] == 'relu':
                print('layer%s: relu'%i)
                aa = self.relu(zz)
                a.append(aa)
                #print(aa)
            else:
                print('layer%s: softmax'%i)
                print(zz)
                aa = self.softmax(zz)
                a.append(aa)
                #print(aa)
            
        return a, z
    
     # leabl one hot encoding
     def one_hot(self, label):
        onehot_encoder = OneHotEncoder(sparse = False)
        label = label.reshape(-1, 1)
        onehot_encoded = onehot_encoder.fit_transform(label)
        return onehot_encoded
    
     # activation function
     def relu(self, a):
        return np.maximum(0, a)
    
     def relu_derivative(self, a):
        relu = lambda x : 1 if x > 0 else 0
        drelu = np.vectorize(relu)(a)
        return drelu
    
     def softmax(self, a):
        print(a)
        exps = np.exp(a - np.max(a ,axis = 1).reshape(-1,1))
        return exps / np.sum(exps,axis = 1)[:, None]
    
     # loss function
     def cross_entropy(self, test, predict):
        return np.mean(-np.sum(test * np.log(predict + 1e-8), axis=1))
    
     # accuracy
     def accuracy(self, test, predict):
        predict_lebel = np.argmax(predict, axis = 1)
        return np.sum((predict_lebel == test) / len(test))
    
     def backward(self, t, a, N, x, z):
        delta = []
        last_a = []
        
        for i in range(len(self.weights) - 1,-1,-1):
            print('gradient w%s' %i)
            if i == 2: # output layer
                delta = a[-1] - t
            else: # hiddlen layer
                f = self.relu_derivative(z[i])
                last_delta = np.dot(delta, self.weights[i + 1].transpose())
                delta = np.multiply(last_delta, f)
            
            if i == 0:
                last_a = x.transpose()
            else:
                last_a = a[i - 1].transpose()
    
            dw = np.dot(last_a, delta)
            #print(dw)
            self.weights[i] = self.weights[i] - self.learning_rate / N * dw
            #print(w[i])
   
     def gradient(self, x, y):   
        loss_list = []
        acc_list = []
        for ep in range(self.epoch):
            print('---------------epoch%s----------'%ep)
            for it in range(self.iteration): 
                print(self.iteration)
                print('---------------iteration%s---------------'%it)
                # batch
                x_batch = x[self.batch_size * it : self.batch_size * (it + 1)]
                y_batch = y[self.batch_size * it : self.batch_size * (it + 1)]
                
                print('\nforward')
                # forward
                forward_output, z = self.forward(x_batch)
                on_hot_lebel = self.one_hot(y_batch)
                
                # loss function
                loss = self.cross_entropy(on_hot_lebel, forward_output[-1])  
                loss_list.append(loss)
                
                # accuarcy
                acc = self.accuracy(y_batch, forward_output[-1])
                acc_list.append(acc)
                
                print('\nbackward')
                # backward
                self.backward(on_hot_lebel, forward_output, len(x_batch), x_batch, z)    
        
        #print(loss_list)
        #print(acc_list)
        return loss_list, acc_list
   
#%% argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config ＜config＞.json", help = "set config")
parser.add_argument("--weight ＜weight＞.npz", help = "set weight")
parser.add_argument("--imgfilelistname ＜imgfilelistname＞.txt", help = "set imgfilelistname")
args = parser.parse_args()    

    
#%% (1)(a) plot
# train
network = FNN(epoch = params['epoch'], batch_size = params['batch_size'], layer = params['nn']
                            , iteration = int(math.ceil(len(x_train) / params['batch_size'])), learning_rate = params['lr']
                            , act = np.array([params['nn'][i]['act'] for i in params['nn'].keys()]))
train_loss, train_acc = network.gradient(x_train, y_train)

# test
network = FNN(epoch = params['epoch'], batch_size = int(len(x_test)/(len(x_train)/params['batch_size'])), layer = params['nn']
                            , iteration = int(math.ceil(len(x_train) / params['batch_size'])), learning_rate = params['lr']
                            , act = np.array([params['nn'][i]['act'] for i in params['nn'].keys()]))
test_loss, test_acc = network.gradient(x_test, y_test)

# plot
plot(train_loss, train_acc, test_loss, test_acc, params['batch_size'])

#%% (1)(b) different batch aize
# train
network = FNN(epoch = params['epoch'], batch_size = 4096, layer = params['nn']
                            , iteration = int(math.ceil(len(x_train) / 4096)), learning_rate = params['lr']
                            , act = np.array([params['nn'][i]['act'] for i in params['nn'].keys()]))
train_loss, train_acc = network.gradient(x_train, y_train)

# test
network = FNN(epoch = params['epoch'], batch_size = int(len(x_test)/(len(x_train)/4096)), layer = params['nn']
                            , iteration = int(math.ceil(len(x_train) / 4096)), learning_rate = params['lr']
                            , act = np.array([params['nn'][i]['act'] for i in params['nn'].keys()]))
test_loss, test_acc = network.gradient(x_test, y_test)

# plot
plot(train_loss, train_acc, test_loss, test_acc)

#%% (1)(c) zero initialization weight
network = FNN(epoch = params['epoch'], batch_size = params['batch_size'], layer = params['nn']
                            , iteration = int(math.ceil(len(x_train) / params['batch_size'])), learning_rate = params['lr']
                            , act = np.array([params['nn'][i]['act'] for i in params['nn'].keys()])
                            , weights = 0)
train_loss, train_acc = network.gradient(x_train, y_train)

# test
network = FNN(epoch = params['epoch'], batch_size = int(len(x_test)/(len(x_train)/params['batch_size'])), layer = params['nn']
                            , iteration = int(math.ceil(len(x_train) / params['batch_size'])), learning_rate = params['lr']
                            , act = np.array([params['nn'][i]['act'] for i in params['nn'].keys()])
                            , weights = 0)
test_loss, test_acc = network.gradient(x_test, y_test)

# plot
plot(train_loss, train_acc, test_loss, test_acc, params['batch_size'])

#%% (2)
weights_list = np.load('weight.npz')
params = json.load(open('config.json',))
network = FNN(epoch = params['epoch'], batch_size = params['batch_size'], layer = params['nn']
                            , iteration = int(math.ceil(len(x_train) / params['batch_size'])), learning_rate = params['lr']
                            , act = np.array([params['nn'][i]['act'] for i in params['nn'].keys()])
                            , weights = weights_list)
train_loss, train_acc = network.gradient(x_train, y_train)

image = open('imgfilelistname.txt', 'r')
test_image = []
for line in image.readlines():
    img = cv2.imread(line.replace('\n',''), 0)
    test_image.append(img)
    
test_image = np.array([i.ravel() for i in (np.array(test_image) / 255)]) 
predict = (network.predict(test_image))
print('predict label:',predict)

np.savetxt('output.txt', predict, fmt="%i", newline="")





