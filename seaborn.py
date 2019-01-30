#%%
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='white',color_codes=True)

iris = pd.read_csv('C:\Python\iris.csv')

iris.head()


sns.jointplot(x='sepal_length',y='sepal_width', data=iris, height=5)
