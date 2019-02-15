#%%
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

np.random.seed(2019)

df = pd.DataFrame({
    'x1' : np.random.chisquare(8,10000),
    'x2' : np.random.normal(5,3,10000),
    'x3' : np.random.normal(-5,5,10000)
})

def standardScaler(x):
    return (x-x.mean())/np.std(x)

def minMaxScaler(x):
    return (x-x.min())/(x.max()-x.min())

def maxAbsScaler(x):
    return (x/np.max(np.abs(x)))

def robustScaler(x):
    return (x-np.median(x))/(np.quantile(x,0.75)-np.quantile(x,0.25))

def practicel():
    np.random.seed(2019)

    df = pd.DataFrame({
        'x1' : np.random.chisquare(8,10000),
        'x2' : np.random.normal(5,3,10000),
        'x3' : np.random.normal(-5,5,10000)
    })


    fig , ax = plt.subplots(nrows=1, ncols=5, figsize=(12,5))

    scaler_list = {
        'standard':standardScaler,
        'minmax':minMaxScaler,
        'maxabs':maxAbsScaler,
        'robust':robustScaler
        }

    ax[0].set_title('Before scaling')
    sns.kdeplot(df['x1'],ax=ax[0])
    sns.kdeplot(df['x2'],ax=ax[0])
    sns.kdeplot(df['x3'],ax=ax[0])

    for i,s in enumerate(scaler_list.items()):
        idx = i+1
        ax[idx].set_title('%s '%(s[0]))
        sns.kdeplot(s[1](df['x1']),ax=ax[idx])
        sns.kdeplot(s[1](df['x2']),ax=ax[idx])
        sns.kdeplot(s[1](df['x3']),ax=ax[idx])

    plt.show()

# 각 scale 별로 식에 따라. 
# standardScaler(x) : 정규 분포로 변환, minMaxScaler(x) : 0~1 사이의 범위가 되로록 변환
# maxAbsScaler(x) : 최대 절대값이 1이 되도록 변환, robustScaler(x) : median, IQR 사용하여 Outlier 영향 최소화 
# 4개의 함수를 생성한 후 for 문으로 각각 그림.  

practicel()

#%% practice2 보스턴 집값. 
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

data = datasets.load_boston()
dfx = pd.DataFrame(data.data,columns=data.feature_names)
dfxo = dfx.drop(['CHAS','RAD'],axis=1)
dfx = dfxo.copy()
dfy = pd.DataFrame(data.target,columns=['MEDV'])




def validateRegression(x,y,comment):
    n = 1000
    avg = 0
    for i in range(n):
        x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,test_size=0.3)
        regr.fit(x_train,y_train)
        avg+=regr.score(x_test,y_test)
    print(comment, avg/n)



regr = LinearRegression()
sigtransform_list = {
    # 'leftshift':np.left_shift,
    # 'rightshift':np.right_shift,
    'none':lambda x : x,
    'leftshift':lambda x: x**3,
    'mildleftshift':lambda x: x**2,
    'log':np.log,
    # 'log2':np.log2,
    'sqrt':np.sqrt
}

#변수분포확인 및 변환/ 변환example 

cols = dfx.columns
plt.cla()

t_i = {k:'none' for k in range(11)}
# t_i[5]='mildleftshift'
# t_i[8]='leftshift'
# t_i[3]='log'
t_i[6]='log'
# t_i[0]='log'
# # t_i[9]='leftshift'
t_i[10]='log'

# data 특징 확인 코드 
fig, axes = plt.subplots(nrows=11,ncols=2,figsize=(5,26))
for i,c in enumerate(cols):       
    fkname = t_i[i]
    axes[i,0].set_title('%s before'%(c))
    sns.kdeplot(dfxo[c],ax=axes[i,0])        
    dfx[c] = sigtransform_list[fkname]((dfxo[c]))
    axes[i,1].set_title('%s after %s'%(c,i))
    sns.kdeplot(dfx[c],ax=axes[i,1])
plt.subplots_adjust(hspace=0.5)
plt.show()

validateRegression(dfxo,dfy,'score before transformation')
validateRegression(dfx,dfy,'score after transformation')


# AGE / PTRATIO right shift(x**3) test
# DIS / CRIM / NOX / LSTAT left shift(log) test
# 결과적으로는 DIS / LSTAT 두 개만 log 취해주면 76.8정도로 best이다. 