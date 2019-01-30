import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('data_viz.csv')

df.descride()

df.innull().sum()

count_classes = pd.value_counts(df['class'], sort=True)
count_classes.plot(kind='bar',rot=0)


#유럽에서의 신용카드 데이터 2일동안 1000건정도 사기가 발생한다.
#결재 금액의 크기에 따라서는 차이가 없다. 

