#%%
import numpy.linalg as NL
import numpy as np

a1_1 = np.array([[1,2],[3,4]])
a1_2 = np.array([[1,2],[3,4],[5,6]])

NL.eig(a1_1) # ev 2개,  크기가 2인 ev2개
# NL.eig(a1_2) # 정방이 아니어서 에러 
NL.svd(a1_1,full_matrices=True) # 2x2 , singular v 2개 2x2
NL.svd(a1_1,full_matrices=False) # 2x2 , singular v 2개 2x2
NL.svd(a1_2,full_matrices=True) # 3x3 , singular v 2개 2x2
NL.svd(a1_2,full_matrices=False) # 3x2 , singular v 2개 2x2


#%% 2 번 20페이지
#x  evalue y eigenvalo

#%% 3번 28페이지


#%% 4번 row 쪼개기 30장 32페이지 

#%% 5번 dot 하면 행렬 곱, 아니면 각 원소의 곱. 


#%%
#6
# #(a)
# a6_1=np.zeros((4,5))
# a6_1 + np.arange(5)
a6_1=np.full((4,5,3),3)

#%%
#(b) ==> matrix
a6_2 = np.full((4,5,3),3)
# a6_2 = np.full((5,5),3)
print(a6_2)
#%%
print(np.cov(a6_2))
a6_2