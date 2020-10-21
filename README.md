## Fixed-num voxelization on point clouds using cython wrapped CUDA/C++
This code provides an API to voxelize input point clouds and outputs the sampled fixed-num point clouds

#### Requirements:
* cython (>=0.16)
* CUDA

YOU need to first ensure you have added environment settings in your ~/.bashrc:

`export PATH="/usr/local/cuda/bin:$PATH"`

`export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"`

#### Install:

To inplace install:

`$ python setup.py build_ext --inplace`
or install
`$ python setup.py install`

to test:

`$ python test.py`



#### API Usage

* **An example**:

```
inputsize = 4096
outsize = 2048
num_y = 120
num_x = 120
num_height = 20
max_length = 1
test_data = test_data.transpose()  #size=[num,3]
test_data = test_data.flatten()
voxelizer = voxelocc.GPUTransformer(test_data, inputsize, max_length, num_x, num_y, num_height, outsize)
voxelizer.transform()
point_t = voxelizer.retreive()
point_t = point_t.reshape(-1,3)  #size=[outsize,3]
```