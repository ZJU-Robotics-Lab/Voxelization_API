## Voxelization on point clouds using cython wrapped CUDA/C++
This code provides an API to voxelize input point clouds and outputs the occupied information of each voxel.

#### Requirements
* cython (>=0.16)
* CUDA

To inplace install:
`$ python setup.py build_ext --inplace`
or install
`$ python setup.py install`

to test:

`$ python test.py`




