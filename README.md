## Voxelization on point cloud 

cython wrapped CUDA/C++ 

This code provide an api to voxelize input point cloud and output the occupied information of each voxel.

To inplace install:

`$ python setup.py build_ext --inplace`

or install

`$ python setup.py install`

to test:

`$ python test.py`

you need a relatively recent version of cython (>=0.16).



