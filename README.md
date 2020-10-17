## Voxelization on point clouds using cython wrapped CUDA/C++
This code provide an api to voxelize input point cloud and output the occupied information of each voxel.

#### Requirements
* cython (>=0.16)
* CUDA

To inplace install:
`$ python setup.py build_ext --inplace`
or install
`$ python setup.py install`

to test:

`$ python test.py`




