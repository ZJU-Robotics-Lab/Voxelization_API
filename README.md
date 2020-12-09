## Voxelization_API

## Voxelization on point clouds using cython wrapped CUDA/C++
This code provides an API to voxelize input point clouds and outputs the occupied information of each voxel.

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

#### More sampling methods can be found in branches
