
#%%
from random import *
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
plt.style.use('classic')

length=10
x = []
y = []
for i in range(length):
    x.append([i])
    y.append([random()*10])
print('x : python (10x1) 2D list','\n',x)
print('y : python (10x1) 2D list','\n',y)

regr = linear_model.LinearRegression()
regr.fit(x,y)

plt.scatter(x,y,color='black')
plt.plot(x, regr.predict(x), color='blue', linewidth=3)
plt.show()

#%%
from random import *
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
plt.style.use('classic')

length=10
x = np.array(range(length)).reshape(length,1)
y = np.array([random()*10 for i in range(length)]).reshape(length,1)
print('x : numpy (10x1) 2D list','\n','array(','\n',x,')')
print('y : numpy (10x1) 2D list','\n','array(','\n',y,')')

regr = linear_model.LinearRegression()
regr.fit(x,y)

plt.scatter(x,y,color='black')
plt.plot(x, regr.predict(x), color='blue', linewidth=3)
plt.show()


#%%
from random import *
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
plt.style.use('classic')

length=10
xy_data=[[round(random()*10,4),round(random()*10,4)]for i in range(length)]
p_data=pd.DataFrame(data=xy_data, columns=('X','Y'))

regr = linear_model.LinearRegression()
regr.fit(p_data.X,p_data.Y)
# pandas는 원래 에러난다. 왜?
# plt.scatter(x,y,color='black')
# plt.plot(x, repr.predict(x), color='blue', linewidth=3)
# plt.show()

#%%
from random import *
import numpy as np
import pandas as pd
from sklearn import linear_model,datasets
import matplotlib.pyplot as plt
plt.style.use('classic')

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:,np.newaxis,2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression(copy_X=0)
regr.fit(diabetes_X_train,diabetes_Y_train)

print('Coefficients : ' , regr.coef_)
print('Intercept : ' , regr.intercept_)

print('MSE : %.2f'%  np.mean((regr.predict(diabetes_X_test)-diabetes_Y_test)**2))

print('Varience score : %.2f'% regr.score(diabetes_X_test,diabetes_Y_test))

plt.scatter(diabetes_X_test,diabetes_Y_test,color = 'black')
plt.plot(diabetes_X_test,regr.predict(diabetes_X_test),color='blue')

plt.xticks(())
plt.yticks(())

#%%
from random import *
import numpy as np
import pandas as pd
from sklearn import linear_model,datasets,neighbors
import matplotlib.pyplot as plt
# import matplotlib.colors.ListedColormap
plt.style.use('classic')

n_neighbors = 15
iris = datasets.load_iris()
X=iris.data[:,:2]
y=iris.target
h=.02

# cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])
plt.scatter(X[:,0],X[:,1],s=100,c=y)


clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X,y)
clf

new_comer= np.array([[3.7,4.5]])
iris_class = clf.predict(new_comer)
print('The iris_class for new_point : ',iris_class)

#색칠하기
x_min , x_max = X[:,0].min()-1 , X[:,0].max()+1
y_min , y_max = X[:,1].min()-1 , X[:,1].max()+1

xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
#41쪽 이어서 해볼 것 

#결정나무 
#%%
from random import *
import numpy as np
import pandas as pd
from sklearn import linear_model,datasets,neighbors,tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pydotplus

length= 10 

x=[]
y=[]
for i in range(length):
    x.append(i)
    y.append(round(random()*10,2))

print('x : python (10x1) 1D array : ','\n',x)
print()
print('y : python (10x1) 1D array : ','\n',y)
print()

xy_data = []
for i in range(length):
    xy_data.append([x[i],y[i]])
print('xy_data : python (10x1) 1D array : ','\n', xy_data)
print()

z_data = [[1],[1],[1],[1],[1],[2],[2],[2],[2],[2]]
print('z_data : python (10x1) 1D array : ','\n', z_data)
print()

clf = tree.DecisionTreeClassifier()
clf.fit(xy_data,z_data)

print(clf.predict([[3,3.01]]))
print(clf.predict(np.array([[7,8.01]])))



