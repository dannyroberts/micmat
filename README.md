# MICMat

MICMat (MIC Matrix) is a module that enables interfacing with Intel's Xeon Phi Coprocessor in a convenient way, directly from pure Python. Some of the features it offers include:

* **Efficient computation**: MICMat is optimized for high performance. Behind the scenes it interfaces with Intel's MKL (BLAS, VSL, VML, and so on), and handles parallelism considerations.
* **Object-oriented development in pure Python**: The MICMat objects are very easy to use, as they are manipulated with syntax very similar to NumPy's.
* **Offload management**: All routines are designed to work either on the host or on the MIC. It is very straightforward to control data transfer between the host and the MIC, and data persistence if fully supported.
* **Automatic memory management**: MICMat offers automatic garbage collection, is at is compatible with the Python garbage collection routines. However, it also offers explicit memory management and reuse, which allows avoiding expensive memory allocation and transfer.

* **Extensive matrix and tensor manipulation routines**: Various matrix and multidimensional array operations are supported and are optimized for performance.



# Installation
## Dependencies

## Compilation


# Usage