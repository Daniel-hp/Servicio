import numpy as np
import matplotlib.pyplot as plt


arr  = np.load("magnitudes14.npy")

print(len(arr[0]))
print( (arr>500).sum() )
print((arr<500).sum())
print((arr==1000).sum())
print(arr.max())
n, bins, patches = plt.hist(arr, 50)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.axis([0, 1000, 0, (arr==0).sum()])
plt.grid(True)
plt.show()