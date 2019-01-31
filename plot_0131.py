#%%
import matplotlib.pyplot as plt

plt.plot([1,2,3],[4,5,6])
plt.xlim([0,4])

plt.show()

#%%
import scipy
from scipy.stats import binom

print(round(binom.pmf(3,5,0.1),4))
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

n1=
n2=
xbar_1=
xbar_2=
s_1=
s_2=


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