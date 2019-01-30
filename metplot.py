import matplotlib.pyplot as plt
import numpy as np

spread = np.random.rand(50)*100
center = np.ones(25)
fliter_high = np.random.rand(10)*100+100
fliter_low = np.random.rand(10)*-100
data = np.concatenate((spread,center,fliter_high,fliter_low))

print(data)

#%%
import matplotlib.pyplot as plt 

plt.figure(1)
plt.subplot(211)
plt.plot([1,2,3])
plt.plot([4,5,6])
plt.figure(2)
plt.plot([4,5,6])

# plt.figure(1)
# plt.subplot(211)
# plt.title('easy as 1,2,3')

plt.clf()

plt.figure(1)
plt.cla()
plt.subplot(211)
plt.title('easy as 1,2,3')

#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('ds_spuar')
plt.imshow(img)
