import voxelocc
import numpy as np
import numpy.testing as npt
import time
import os
import numpy.testing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_pc_file(filename):
    # returns Nx3 matrix
    pc = np.fromfile(os.path.join("./", filename), dtype=np.float64)

    if(pc.shape[0] != 4096*3):
        print("Error in pointcloud shape")
        return np.array([])

    pc = np.reshape(pc,(pc.shape[0]//3, 3))    
    return pc

#### Test
test_data = load_pc_file("test.bin")
print(test_data.shape)
x = test_data[...,0]
y = test_data[...,1]
z = test_data[...,2]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
plt.show()
plt.pause(0.1)
plt.close()

test_data = test_data.astype(np.float32)
# test_data = test_data[np.newaxis,:,...]

#### Settings
num_points = 4096
size = num_points
outsize = 2048
num_y = 120
num_x = 40
num_height = 20
max_length = 1  # max_length in xy direction (same for x and y)
enough_large = 1 # Num points in a voxel, no use for occupied information

#### Usage for voxelization
test_data = test_data.transpose()
test_data = test_data.flatten()
voxelizer = voxelocc.GPUTransformer(test_data, size, max_length, num_x, num_y, num_height, outsize)
voxelizer.transform()
point_t = voxelizer.retreive()
point_t = point_t.reshape(-1,3)

#### Test fix-num sampling
x = point_t[...,0]
y = point_t[...,1]
z = point_t[...,2]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
plt.show()
plt.pause(0.1)
plt.close()

print(point_t.shape)

