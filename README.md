# mSTOMP-GPU
This is a GPU implementation of the mSTAMP STOMP algorithm. mSTOMP takes multiple time series as input and computes the matrix profile for a particular window size and number of dimensions. You can read more at the [Matrix Profile Homepage](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
The dimensions are read from the input file one after another. The output is done in the same way: matrix profiles for k=1..ndim are output one after another

* Test run with example data (taken from the mSTAMP toy data sample):
  `mSTOMP-GPU.exe 3 30 toy_data30.txt 3 toy_out.txt toy_outi.txt`

For additional features and better performance you should use [SCAMP](http://github.com/zpzim/SCAMP) (doesn't support multi-STAMP)
# Environment
This base project requires:
 * At least version 9.0 of the CUDA toolkit available [here](https://developer.nvidia.com/cuda-toolkit).
 * An NVIDIA GPU with CUDA support is also required. You can find a list of CUDA compatible GPUs [here](https://developer.nvidia.com/cuda-gpus)
 * Currently builds under linux with the Makefile. 
 * Should compile under windows, but untested. 
# Usage
* Edit the Makefile
  * Volta is supported by default, but if needed set the value of ARCH to correspond to the compute capability of your GPU.
    * "-gencode=arch=compute_code,code=sm_code" where code corresponds to the compute capability or arch you wish to add.
  * Make sure CUDA_DIRECTORY corresponds to the location where cuda is installed on your system. This is usually `/usr/local/cuda-(VERSION)/` on linux
  * Also, by default the kernel parameters are optimized for volta only, if you are building for Pascal or earlier, please tune the variables TILE_HEIGHT_ADJUSTMENT and UNROLL_COUNT in STOMP.cu accordingly
  * Some suggested parameters for different architectures are provided in the comments
* `make`
* `STOMP window_size input_file_path output_matrix_profile_path output_indexes_path (Optional: list of device numbers that you want to run on)`
* Example:
* `STOMP 1024 SampleInput/randomlist128K.txt profile.txt index.txt 0 2`
* By default, if no devices are specified, STOMP will run on all available devices

* There are Visual Studio 2013 project files for CUDA 9 included
