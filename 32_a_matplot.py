
# In[1]
#1. 데이터 읽기 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('C:\Python\data_viz.csv')

# print(df.head())

# print('describe \n',df.describe())
# print('isnull \n',df.isnull().sum())
# # 알아보기 어려움, null값 없음 

# count_classes = pd.value_counts(df['Class'], sort=True)
# count_classes.plot(kind='bar',rot=0)
#class는 0,1로 이루어져 있고 1이 매우 적다. 

#유럽에서의 신용카드 데이터 2일동안 1000건정도 사기가 발생한다.
#결재 금액의 크기에 따라서는 차이가 없다. 

# In[2]
col=df.columns
print(col)

# sns.jointplot(x='Time',y='V1', data=df, size=5)
for i in col:
    sns.FacetGrid(df, hue='Class', size=3 )\
        .map(plt.scatter, 'Time',i)\
        .add_legend()


# In[3]
for i in col:
    sns.FacetGrid(df, hue='Class', size=3 )\
        .map(plt.scatter, 'Amount',i)\
        .add_legend()

