#%%
import numpy.linalg as la
import numpy as np

w , v = la.eig(np.array([[3,0],[8,-1]]))

w, v # eigenvalues , eigenvector

#%% 
l=np.diag(w)
i=la.inv(v)

np.dot(np.dot(v,l),i)



#%% SVD
import numpy as np
np.set_printoptions(precision=0)

A = np.array([[4,2,3,5,1],[0,3,0,4,2,],[5,4,3,3,0],[0,0,5,5,2],[5,0,0,5,0]])

print(A)

#SVD
U , s , V = np.linalg.svd(A, full_matrices= True)

#Reconstruct
S3=np.zeros((5,5))
S3[:3,:3]=np.diag(s[:3])

S4=np.zeros((5,5))
S4[:4,:4]=np.diag(s[:4])

S5=np.zeros((5,5))
S5[:5,:5]=np.diag(s[:5])

P = np.dot(U,np.dot(S3,V))
P4 = np.dot(U,np.dot(S4,V))
P5 = np.dot(U,np.dot(S5,V))


print(P)
print(P4)
print(P5)

