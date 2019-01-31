
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

repr = linear_model.LinearRegression()
repr.fit(x,y)

plt.scatter(x,y,color='black')
plt.plot(x, repr.predict(x), color='blue', linewidth=3)
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

repr = linear_model.LinearRegression()
repr.fit(x,y)

plt.scatter(x,y,color='black')
plt.plot(x, repr.predict(x), color='blue', linewidth=3)
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

repr = linear_model.LinearRegression()
repr.fit(p_data.X,p_data.Y)
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
