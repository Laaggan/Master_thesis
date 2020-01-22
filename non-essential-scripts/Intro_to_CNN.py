import numpy as np
from old_functionality import *
import matplotlib.pyplot as plt

# Kernel 1 is a vertical line detector
kernel1 = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
kernel1 = np.array(kernel1)

# Kernel 2 is a horizontal line detector
kernel2 = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
kernel2 = np.array(kernel2)

# Create an image with a vertical line
n = 5
input_data = np.zeros([5, 5])
input_data[:, 3] = 1

def Convolution(input_data, kernel):
    # Size of kernel and input data, assuming square matrices
    n = input_data.shape[0]
    m = kernel1.shape[0]

    # d is the size of feature map with stride 1
    d = n - m + 1
    feature_map = np.zeros([d, d])

    # This loop works for stride 1 and square input data
    for i in range(d):
        for j in range(d):
            feature_map[i, j] = np.sum(np.multiply(input_data[i:(i+m), j:(j+m)], kernel))
    return feature_map

print(input_data)
# We see strong activations for vertical line
print(Convolution(input_data, kernel1))
# We see weaker activations for horizontal line
print(Convolution(input_data, kernel2))

#plot vertical kernel conved with input image
k1conim=Convolution(input_data, kernel2)
plt.imshow(k1conim)
plt.show()