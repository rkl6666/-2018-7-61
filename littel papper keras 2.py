# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 11:13:32 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:16:17 2018

@author: Administrator
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
#from keras.optimizers import SGD,Adma
from keras.regularizers import l2
from sklearn.utils import shuffle 
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
import keras.optimizers 

#加载keras模块
#from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt
#matplotlib inline

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r') #label='train acc')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b')# label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.show()

#创建一个实例LossHistory
history = LossHistory()


#数据集
data=np.load('X_new.npy')
Y=np.load('y.npy')
X1=data[0:41243]
Y1=Y[0:41243]
X2=data[41243:58919]
Y2=Y[41243:58919]





'''
df=np.load('shujuji.npy')
min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(df)
df = shuffle(df)
Xxunlian=df[0:35000,0:26]
Xceshi=df[35000:58919,0:26]
Yxunlian=df[0:35000,26:27]
Yceshi=df[35000:58919,26:27]
print(Xxunlian.shape)
'''



#转one-hot标签
Yw1=np_utils.to_categorical(Y1,num_classes=2)
Yw2=np_utils.to_categorical(Y2,num_classes=2)
#Yceshi=np_utils.to_categorical(Yceshi,num_classes=2)
#创建模型,输入12个神经元,输出10个神经元

model=Sequential([Dense(units=50,input_dim=12,bias='one',activation='relu',kernel_regularizer=l2(0.003)),
                  Dropout(0.00),
                  Dense(units=50,bias='one',activation='relu',kernel_regularizer=l2(0.003)),
                  Dropout(0.00),
                  Dense(units=50,bias='one',activation='relu',kernel_regularizer=l2(0.003)),
 #                 Dense(units=100,bias='one',activation='tanh',kernel_regularizer=l2(0.003)),
                  Dropout(0.00),
                  Dense(units=2,bias='one',activation='softmax',kernel_regularizer=l2(0.003))])
'''
model = Sequential()
model.add(Dense(units=30,input_dim=12,bias='one',activation='relu',kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add( Dense(units=30,bias='one',activation='relu',kernel_regularizer=l2(0.003)))
model.add(Activation('relu'))
'''
#定义epoach次数
ep=200
#定义优化器
#使用adma算法
adma=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0)
#定义loss func
model.compile(optimizer=adma,loss='categorical_crossentropy',metrics=['binary_accuracy'])
#训练模型
History=model.fit(X1,Yw1,batch_size=300,epochs=ep,validation_split=0.2,callbacks=[history])

#print(metrics.val_f1s)
#画图
#history.loss_plot('epoch')


#测试模型
loss,accuracy=model.evaluate(X2,Yw2)

print('test loss:',loss)
print('test accuracy:',accuracy)

#print('train loss:',loss)
#print('train accuracy:',accuracy)
Y_pred = model.predict(X2)
#print(Y_pred[1])

list1=[]
for i in Y_pred:
    if i[0]>i[1]:
        list1.append(0)
    if i[0]<i[1]:
        list1.append(1)
print(len(list1))
    

from sklearn.metrics import precision_score, recall_score, f1_score ,accuracy_score
    #打印出三个指标
p = precision_score(Y2,list1, average='binary')
#P.append(p)
r = recall_score(Y2,list1, average='binary')
#R.append(r)
f1score = f1_score(Y2,list1, average='binary')
#F.append(f1score)
acc=accuracy_score(Y2,list1)
#ACC.append(acc)
#scores = precision_recall_fscore_support(dfd2[:,15:16],data_predict)
print (p,r,f1score,acc)

#打印历史参数
ls=History.history
#print(ls)
valoss=ls.get('val_loss')
valacc=ls.get('val_binary_accuracy')
loss1=ls.get('loss')
bacc=ls.get('binary_accuracy')

#print(valoss)
epw=range(0,ep)
listep=list(epw)
#画图部分
import matplotlib.pyplot as plt 
plt.figure(1)
plt.plot(listep,valoss,label='val_loss')
plt.plot(listep,loss1,label='loss')
plt.xlabel('epoach') 
plt.ylabel('loss')
plt.title('curve of loss')
plt.legend()

plt.figure(2)
plt.plot(listep,valacc,label='val_binary_accuracy')
plt.plot(listep,bacc,label='accuracy')
plt.xlabel('epoach') 
plt.ylabel('accuracy')
plt.title('curve of accuracy') 
plt.legend()
plt.show()