#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import random


# In[47]:



def rand_samples(m,b,num,rand_param):
    x_axis,y_axis,labels=np.array([]), np.array([]), np.array([])
    if(m>=0):
        c=1
    else:
        c=-1
    pos_num = int(num / 2)
    neg_num = num-pos_num
    
    for state,num in[['pos', pos_num], ['neg', neg_num]]:
        x = np.random.randint(0, rand_param,num)
        r = np.random.randint(1, rand_param,num)
        y_line=m*x+b
        if state == 'pos':
            y=m*x+b-(r*c)
            for i in range(len(y)):
                for j in range(len(y_line)):
                    if(y[i]==y_line[j]):
                        y-=1
            labels = np.append(labels, np.ones(num, dtype=int))
        else:
            y=m*x+b+(r*c)
            for i in range(len(y)):
                for j in range(len(y_line)):
                    if(y[i]==y_line[j]):
                        y+=1
            labels = np.append(labels,-1*np.ones(num, dtype=int))
       
                       
        x_axis=np.append(x_axis,x)
        y_axis=np.append(y_axis,y)
    return x_axis,y_axis,labels
            
            


# In[48]:


m,b=2,1
num=30
rand_param=30
pos_num=int(num/2)
x=np.arange(rand_param+1)
y=m*x+b
plt.plot(x,y)#(0,1)->(30,62)
x_axis,y_axis,labels=rand_samples(m,b,num,rand_param)
plt.plot(x_axis[:pos_num],y_axis[:pos_num],'o',color="blue")
plt.plot(x_axis[pos_num:],y_axis[pos_num:],'o',color="red")


# In[49]:


df=pd.DataFrame({'x':x_axis,'y':y_axis,'labels':labels},dtype="float64")
dataset=df.values
print(dataset[0][2])


# In[54]:


def Display(dataset,w0,w1,b): 
    plt.scatter(dataset[:pos_num,0], dataset[:pos_num,1], color='blue', marker='o', label='Positive') 
    plt.scatter(dataset[pos_num:,0], dataset[pos_num:,1], color='red', marker='o', label='Negative')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.title('Scatter')
    #法向量=w2x+w1y-(w2x0+w1y)=0
   #y1=(w0-b)/w1
    #y2=-1*(w0+b)/w1
    plt.plot([-1,1],[(w0-b)/w1,-1*(w0+b)/w1],'g')
    #y1=(w0-b)/w1
    #y2=-1*(w0+b)/w1
    #x=np.arange(30+1)
   # a=-w0/w1
    #b1=-c/w1
   # y=a*x #y=a*x+b1
    #plt.plot(x,y,'y')
   # print("from 0,{:6f} to 30,{:6f}".format(y1,y2))
    plt.show()
def PLA():
    starttime = time.time() 
    W = np.zeros(2) #[0,0]原點
    W0 = 0           #w0=0
    count = 0
    while True:
        count += 1
        cease = True
        for i in range(0,len(dataset)):
            xn = dataset[i][:-1]      #  x = dataset[i][:-1]
                                        #X = np.array(x)
                                       #Y = np.dot(W,X) + b
            Xn = np.array(xn)
            Y = np.dot(W,Xn) + W0
            if np.sign(Y) == np.sign(dataset[i][2]): #Y>=0 
                continue
            else:
                cease = False
                W = W + (dataset[i][-1]) * Xn
                W0 = W0 + dataset[i][-1]
        if cease:
            break
    endtime = time.time()
    dtime = endtime - starttime
    print("W:",W)
    print("count:",count)
    print("time: %.8s s" % dtime)
    print("count:",count)
    print(W0)
    Display(dataset,W[0],W[1],W0)
    return W
def main():
    W =  PLA()
if __name__ == '__main__':
    main()


# In[51]:


print(dataset[0][:2])
print(dataset[0][:])


# In[ ]:




