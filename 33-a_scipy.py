#1
#%%
import numpy as np
import scipy as stats

samples = np.random.normal(size=500)

print(np.median(samples))

print(np.std(samples))

print(stats.scoreatpercentile(samples,80))

loc,std= stats.norm.fit(samples)
print(loc)
print(std)



# #(a)
# #2
# from scipy import linalg
# import numpy as np

# arr=np.array([1,3,5],[2,4,6],[6,5,8])

# print(linalg.det(arr))

# print(linalg.inv(arr))

# arr2=np.array([1,2,3,4],[3,8,5,2],[4,3,6,2])
# #스퀘어가 아니라서 그렇지롱 

# #4는 작성 안해도 된다. 

# #5
# import numpy as np

# a = np.array([2,2,2],[4,7,7],[6,18,22])
# P,L,U = linalg.lu(A)

# print(str(L))
# print(str(U))
# print(str(np.matmul(P,np.matmul(L,U))))

# #6
# np.random.seed(0)

# x_data=np.linspace(-5,5,num=50)
# y_data= 4*np.cos(2*x_data)+np.random.normal(size=50)

# import matplotlib.pyplot as plt
# plt.scatter(x_data,y_data)
# plt.show()

# #7
# from scipy import optimize
# def test_func(x,a,b):
#     return a*np.cos(b*x)
# param,prams_covariance = optimize.curve_fit(test_func,x_data,y_data,p0=[2,2])

# print(param)


# #8 
# import scipy as stats
# class1=[65.9,53.6,57.3,59.3,63.8,59.2,64.2,75.0,62.9]
# class2=[76.3,82.1,73.3,69.3,59.9,72.1,59.1,86.8,78.1]

# result = stats.ttest_ind(class1,class2)

# print('독립표본 t 검정 결과: %.4f, pvalue=%.3f'%(result))