#%%
import matplotlib.pyplot as plt

plt.plot([1,2,3],[4,5,6])
plt.xlim([0,4])

plt.show()

#%%
import scipy
from scipy.stats import binom

print(round(binom.pmf(3.5,0.1),4))
print(round(binom.cdf(k=5,n=20,p=0.1),4))
print(round(1-binom.cdf(k=3,n=20,p=0.1),4))
print()
print(binom.ppf(q=0.1875,n=5,p=0.5))


#%%
import matplotlib.pyplot as plt
def showplot():
    plt.plot(x,fx,marker='o',linestyles='')
    plt.vlines(x,0,fx)

#%%
from scipy.stats import uniform
print(uniform.pdf(x=1,loc=0,scale=2))

#%%
from scipy.stats import norm
round(norm.pdf(x=0,scale=1),3)

#%%
import numpy as np

x=np.arange(-5,5.1,0.1)

fx=norm.pdf(x=x,loc=0,scale=1)
def showplot():
    plt.plot(x,fx)
    plt.plot('x')
    plt.plot('fx')
    plt.title('pdf of standard normal dist')
    plt.ylim([-0.001,0.40001])
showplot()

#%%
print(norm.cdf(x=2500,loc=2000,scale=200))
print(round(1-norm.cdf(x=1800,loc=2000,scale=200),3))
print(round(norm.ppf(q=0.98,loc=100,scale=15),3))


#%%
import numpy as np
import scipy
from scipy.stats import binom,norm
def showplot():
    plt.subplot(131)
    plt.xlim(-3,15)
    plt.plot(range(11),binom.pmf(range(11),n=10,p=0.1),marker='o')
    plt.plot(range(-3,15),norm.pdf(range(-3,15),loc=1,scale=np.sqrt(0.9)),color='black')
    
    plt.subplot(132)
    plt.xlim(-3,15)
    plt.plot(range(16),binom.pmf(range(16),n=30,p=0.1),marker='o')
    plt.plot(range(-3,15),norm.pdf(range(-3,15),loc=3,scale=np.sqrt(2.7)),color='black')
    
    plt.subplot(133)
    plt.xlim(-3,15)
    plt.plot(range(26),binom.pmf(range(26),n=50,p=0.1),marker='o')
    plt.plot(range(-3,15),norm.pdf(range(-3,15),loc=5,scale=np.sqrt(4.5)),color='black')

showplot()


#%%
import numpy as np
import scipy
from scipy.stats import binom,norm,t

x=np.arange(-5,5.1,0.1)

plt.style.use('classic')

plt.plot(x,t.pdf(x,3),color='black',label='3')
plt.plot(x,t.pdf(x,6),color='red',label='6')
plt.plot(x,t.pdf(x,9),color='blue',label='9')
plt.plot(x,t.pdf(x,20),color='orange',label='20')
plt.plot(x,norm.pdf(x,0,1),color='green',label='normal')
plt.legend(loc='upper right')

#%%
import numpy as np
import scipy
from scipy.stats import binom,norm,t
n=49; sigma=30; xbar=157.02; d=5; alpha=0.05

z_alpha=norm.ppf(1-alpha/2)
c_i=np.array([round(xbar-z_alpha*sigma/np.sqrt(n),3),round(xbar+z_alpha*sigma/np.sqrt(n),3)])

print(c_i)

min_n=(z_alpha*sigma/d)**2
print(np.ceil(min_n))

#%%
import numpy as np
import scipy
from scipy.stats import binom,norm,t
n=75; sigma=0.0015; xbar=0.310; d=0.0005; alpha=0.05

z_alpha=norm.ppf(1-alpha/2)
c_i=np.array([round(xbar-z_alpha*sigma/np.sqrt(n),5),round(xbar+z_alpha*sigma/np.sqrt(n),5)])

print(c_i)

min_n=(z_alpha*sigma/d)**2
print(np.ceil(min_n))

#%%
import numpy as np
import scipy
from scipy.stats import binom,norm,t

n=75; sigma=5.083; xbar=27.75; alpha=0.01

t_alpha=t.ppf(1-alpha/2,n-1)
print(round(t_alpha,3))

c_i=np.array([round(xbar-t_alpha*sigma/np.sqrt(n),3),round(xbar+t_alpha*sigma/np.sqrt(n),3)])
print(c_i)

#%%
import numpy as np
import scipy
from scipy.stats import binom,norm,t

samples=np.array([53.0,51.5,47.0,54.5,44.0,53.0,45.5,56.0,45.5])
xbar=np.mean(samples)
n=9;  alpha=0.05

sigma=np.std(samples)
s=sigma*np.sqrt(n/8)



print(xbar)

c_i=np.array([round(xbar-t.ppf(1-alpha/2,n-1)*s/np.sqrt(n),3),round(xbar+t.ppf(1-alpha/2,n-1)*s/np.sqrt(n),3)])
print(c_i)


sigma=4
z_alpha=norm.ppf(1-alpha/2)
print(round(z_alpha,3))
c_i=np.array([round(xbar-z_alpha*sigma/np.sqrt(n),5),round(xbar+z_alpha*sigma/np.sqrt(n),5)])

print(c_i)


#%%
#파이썬에서는 단측가설을 주지않는다. 
#대칭분포를 따르는 모집단에 대한 검정에서 단측가설의 유의 확률은 양측가설의 유의확률의 1/2
#numpy.std(data, ddof = 1) 하면 (n-ddof) 비 편향 추정량을 계산한다고 하네요 
# st.tstd(cow_d) 요거 써도 됨. 

import numpy as np
import scipy.stats as st

bulb = np.array([2000,1975,1900,2000,1950,1950,1850,2100,1975])
n=9
tt=st.ttest_1samp(bulb,1950)

print(round(tt.statistic,3))

m=np.mean(bulb)

s=np.std(bulb)*np.sqrt(n/(n-1))
print(round((m-1950)/(s/np.sqrt(n)),3))


#강의자료 51쪽

#%%

import numpy as np
from scipy.stats import binom,norm,t
import scipy.stats as st

sc_xbar=0.62; sc_s=0.11; n=120

t_0=(sc_xbar-0.6)/(sc_s/np.sqrt(n))

print(round(t.ppf(0.975,n-1),3))

print(round(norm.ppf(0.975),3))

print(round(2*(1-t.cdf(t_0,n-1)),3))


#%%

import numpy as np
from scipy.stats import binom,norm,t
import scipy.stats as st

k_xbar=1.02; k_sigma=1.196; n=10; alpha=0.05

t_1=(k_xbar-0)/(k_sigma/np.sqrt(n))
print(round(t_1,3))
print(round(t.ppf(1-alpha,n-1),3))
print(round(1-t.cdf(t_1,n-1),3))



#%%
import numpy as np
from scipy.stats import binom,norm,t
import scipy.stats as st

n=6 ; sigma_d=0.443 ; dbar=0.2 ; alpha=0.05
x0=np.array([0.8,-0.2,0.6,-0.3,0.3])
s=np.std(x0)*np.sqrt(n/(n-1))

print(s)
print(np.mean(x0))
c_i=np.array([round(dbar-t.ppf(1-alpha/2,n-1)*sigma_d/np.sqrt(n),3),round(dbar+t.ppf(1-alpha/2,n-1)*sigma_d/np.sqrt(n),3)])
print(c_i)


#%%
import numpy as np
from scipy.stats import binom,norm,t
import scipy.stats as st


cow_1 = np.array([24.7,46.1,18.5,29.5,26.3,33.9,23.1,20.7,18.0,19.3,23.0])
cow_2 = np.array([12.4,14.1,7.6,9.5,19.7,10.6,9.1,11.2,13.1,8.3,15.0])
cow_d = cow_1 - cow_2
n=11

print(round(t.ppf(0.99,10),3))

cow_tt = st.ttest_rel(cow_1,cow_2)
print(round(cow_tt.statistic,3))
d=np.mean(cow_d)

s=np.std(cow_d)*np.sqrt(n/(n-1))

print(round(d/(s/np.sqrt(n)),3))

print(round(cow_tt.pvalue/2,4))

#%%
import numpy as np
from scipy.stats import binom,norm,t
import scipy.stats as st

x_1=np.array([19.54,14.47,16.00,24.83,26.39,11.49])
x_2=np.array([15.95,25.89,20.53,15.52,14.18,16.00])

n1=6
n2=6
s_p=np.sqrt(((n1-1)*np.var(x_1)+(n2-1)*np.var(x_2))/(n1+n2-2))
tt=st.ttest_ind(x_1,x_2,equal_var=True)

print(round(tt.statistic,3))
print(round(tt.pvalue,3))



#%%
import numpy as np
from scipy.stats import binom,norm,t
import scipy.stats as st

x_1
x_2

#검정 통계량 
tt=st.ttest_ind(x_1,x_2,equal_var=False)

print(round(tt.statistic,3))
print(round(tt.pvalue/2,3))

# n1=
# n2=
# xbar_1=
# xbar_2=
# s_1=
# s_2=


#%%
import numpy as np
from scipy.stats import binom,norm,t
import scipy.stats as st

alpha = 0.05
n1=12
n2=10
x_1 = np.array([8.2,8.3,8.4,9.3,8.3,7.0,7.8,7.9,7.5,9.5,6.0,7.6])
x_2 = np.array([8.2,7.0,6.5,8.2,6.4,8.2,6.7,7.6,5.3,6.8])

xbar_1=np.mean(x_1)
xbar_2=np.mean(x_2)
var_1=np.var(x_1)*(n1/(n1-1))
var_2=np.var(x_2)*(n2/(n2-1))

print(round(xbar_1-xbar_2)/np.sqrt(var_1/n1+var_2/n2),3)



#t 분포를 정의 
ct=st.ttest_ind(x_1,x_2,equal_var=False)
#검정통계량
print(round(tt.statistic,3))

df=((var_1/n1+var_2/n2)**2/(((var_1/n1)**2/(n1-1)+(var_2/n2)**2/(n2-1))))
#유의확률
print(round(2*1-t.cdf(ct.statistic,df),3))
print(round(ct.pvalue/2,3))



#%%
from random import *
import numpy as np
import pandas as pd
from sklearn import linear_model,datasets,neighbors,tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pydotplus
from scipy.stats import chi2

x = np.array([226,228,226,225,232,228,229,227,225,230])

sigma=1.5
n=10
alpha=0.05

sdx = np.std(x)*np.sqrt(n/(n-1))

chisq= (n-1)*(sdx**2)/(sigma**2)
print(round(chisq,3))

#기각역의 경계값
print(round(chi2.ppf(1-alpha,n-1),3))

#유의확률/ 기각 가능
print(round(1-chi2.cdf(chisq,n-1),3))

#신뢰구간
print(np.array([round(chisq*(sigma*2)/chi2.ppf(1-alpha,n-1),3),round(chisq*(sigma*2)/chi2.ppf(alpha,n-1),3)]))

#%%
from random import *
import numpy as np
import pandas as pd
from sklearn import linear_model,datasets,neighbors,tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pydotplus
from scipy.stats import chi2,f

m1=171.74
m2=207.92
n1=167
n2=130
sigma1=122.356
sigma2=123.845
alpha=0.05

var_ratio= sigma1**2/sigma2**2
print(round(var_ratio,3))

print(round(1/f.ppf(1-alpha/2,n2-1,n1-1),3),round(f.ppf(1-alpha/2,n1-1,n2-1),3))

#%%
from random import *
import numpy as np
import pandas as pd
from sklearn import linear_model,datasets,neighbors,tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pydotplus
from scipy.stats import chi2,f

#실습 1번 
#%%
import scipy
from scipy.stats import bernoulli,t
import pandas as pd
import numpy as np
import scipy.stats as st
alpah=0.05
n=100
p=0.5

np.random.seed(0)
x=bernoulli.rvs(p,size=n)
print(np.count_nonzero(x))

t1= st.ttest_1samp(x,p)
print(round(t1.statistic,3))

print(round((np.mean(x)-p)/((np.std(x)*np.sqrt(n/(n-1))/np.sqrt(100))),3))

print(round(t1.pvalue,3))
print(round(2*(t.cdf(t1.statistic,n-1)),3))

#실습 1번 
#%%
import scipy
from scipy.stats import bernoulli,t
import pandas as pd
import numpy as np
import scipy.stats as st
alpah=0.05
n=100
p1=0.35
p2=0.5

np.random.seed(0)
x=bernoulli.rvs(p1,size=n)
print(np.count_nonzero(x))

t1= st.ttest_1samp(x,p1)
print(round(t1.statistic,4))

print(round((np.mean(x)-p1)/((np.std(x)*np.sqrt(n/(n-1))/np.sqrt(100))),4))

print(round(t1.pvalue,4))
print(round(2*(t.cdf(t1.statistic,n-1)),4))

#%%
import seaborn as sns
tips = sns.load_dataset("tips")
tips

#%%
#실습예제 4
from scipy.stats import f

a = np.array([175, 168, 168, 190, 156, 181, 182, 175, 174, 179])
b = np.array([185, 169, 173, 173, 188, 186, 175, 174, 179, 180])

var_ratio = np.var(a)/np.var(b)

print(round(var_ratio,3))# 2.103이 기각역의 경계 안쪽에 들어오니까 기각못함(작거나 커야한다.)

print(round(1/f.ppf(0.975,9,9),3),round(f.ppf(0.975,9,9),3))
# 기각역의 경계

print(round(2*(1-f.cdf(var_ratio,9,9)),3))
# 양측 가설의 유의확률
# 귀무가설을 기각할 수 없다. 따라서 등분간 가정이 만족한다.


S_p=np.sqrt((10*np.var(a)+10*np.var(b))/(10+10-2)) 
# Pooled standard deviation

tt=st.ttest_ind(a,b,equal_var=True) 
print(round(tt.statistic,3))
#검정통계량 - 함수 

t1=(np.mean(a)-np.mean(b))/S_p/np.sqrt(1/10+1/10)
print(round(t1,3))
#검정통계량 - 식

print(round(tt.pvalue,3))
# 유의확률 - 함수

print(round(2*t.cdf(t1,10+10-2),3))
# 유의확률 - 식 .356이 0.5보다 작으니까. 기간 못한다. 

# 귀무가설을 기각할 수 없다.

#%%
#실습예제 5
mid = np.array([16, 20, 21, 22, 23, 22, 27, 25, 27, 28])
final = np.array([19, 22, 24, 24, 25, 25, 26, 26, 28, 32])

tt=st.ttest_rel(mid,final)
print(round(tt.statistic,3))
# ttest_rel을 사용한 검정통계량 

d=np.mean(mid-final)
s=np.std(mid-final)*np.sqrt(10/9)

print(round(d/(s/np.sqrt(10)),3))
# 식을 이용하여 계산한 검정통계량. 둘의 결과는 같다.

print(round(tt.pvalue,3))
# 유의확률 함수
print(round(2*t.cdf(d/(s/np.sqrt(10)),9),3))
# 유의확률 식
         
# 유의확률. 귀무가설을 기각할 수 있다. 과외를 받은 전과 이후의 평균 성적 차이는 통계적으로 유의미하다고 할 수 있다.


#%%
#실습예제 6

mid = np.array([16, 20, 21, 22, 23, 22, 27, 25, 27, 28])
final = np.array([19, 22, 24, 24, 25, 25, 26, 26, 28, 32])

np.mean(final)
tt1=st.ttest_1samp(final,24)
print(round(tt1.statistic,3))
# ttest_1samp를 이용한 검정통계량


s1=np.std(final)*np.sqrt(10/9)
# 표본표준편차 - 식 사용을 위해서. 

print(round((np.mean(final)-24)/(s1/np.sqrt(10)),3))
# 식을 이용해 직접 계산한 검정통계량. 둘의 값은 같다.

print(round(tt1.pvalue/2,3))
#유의확률 - 함수 
print(round(1-t.cdf((np.mean(final)-24)/(s1/np.sqrt(10)),9),3))
# 유의확률 - 식 사용. cdf - 분포의 역함수  
# 귀무가설을 기각할 수 없다. 따라서 95% 신뢰수준에서 학생들의 기말고사 성적은 24점보다 높다고 할 수 없다.

#%%
from sklearn import datasets
import pandas as pd


iris=datasets.load_iris()
iris.data

df=pd.DataFrame(iris.data)
df

print(iris.feature_names)
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df

print(iris.target_names)
print(iris.target)
df['species']=np.array([iris.target_names[i] for i in iris.target])
df

df1=df[0:100]
setosa=df1[0:50]
versicolor=df1[50:100]

print(np.mean(setosa))
print(np.mean(versicolor))


#1 등분산 검정 


from scipy.stats import f

s=np.array(setosa['sepal length (cm)'])
v=np.array(versicolor['sepal length (cm)'])

var_ratio = np.var(s)/np.var(v)
print(round(var_ratio,3))

print(round(1/f.ppf(0.975,49,49),3),round(f.ppf(0.975,49,49),3))
# 기각역의 경계

print(round(2*f.cdf(var_ratio,49,49),4))
# 양측 가설의 유의확률
# 귀무가설을 기각할 수 있다. 따라서 등분간 가정이 만족하지 않는다고 할 수 있다.


#2 독립 이분산 검정 

print(np.mean(setosa['sepal length (cm)']))
print(np.mean(versicolor['sepal length (cm)']))

s=np.array(setosa['sepal length (cm)'])
v=np.array(versicolor['sepal length (cm)'])

tt2=st.ttest_ind(s,v,equal_var=False)
print(round(tt2.statistic,3))

df=((np.var(s)*(50/49)/50+np.var(v)*(50/49)/50)**2)/((np.var(s)*(50/49)/50)**2/49+(np.var(v)*(50/49)/50)**2/49)
print(2*(t.cdf(tt2.statistic,df)))

print(tt2.pvalue)
#유의확률. 귀무가설을 기각할 수 있다. 따라서 두 종사이의 sepal.length에 차이가 있다고 할 수 있다.s