# MICMat

MICMat (MIC Matrix) is a module that enables interfacing with Intel's Xeon Phi Coprocessor in a convenient way, directly from pure Python. Some of the features it offers include:

* **Efficient computation**: MICMat is optimized for high performance. Behind the scenes it interfaces with Intel's MKL (BLAS, VSL, VML, and so on), and handles parallelism considerations.
* **Object-oriented development in pure Python**: The MICMat objects are very easy to use, as they are manipulated with syntax very similar to NumPy's.
* **Offload management**: All routines are designed to work either on the host or on the MIC. It is very straightforward to control data transfer between the host and the MIC, and data persistence if fully supported.
* **Automatic memory management**: MICMat offers automatic garbage collection, is at is compatible with the Python garbage collection routines. However, it also offers explicit memory management and reuse, which allows avoiding expensive memory allocation and transfer.

* **Extensive matrix and tensor manipulation routines**: Various matrix and multidimensional array operations are supported and are optimized for performance.



# Installation
## Dependencies
MICMat requires:

* Python 2.7. The [Anaconda](http://continuum.io/downloads) distribution is recommended
* Intel Composer XE 2013 SP 1


## Compilation
From within the micmat sub-directory, build the package:  
```    
. micmat_build.sh
```

## Import into Python
In order import MICMat into Python and to maximize performance on Xeon Phi, various environment variables and paths must be configured in advance. These are all automatically set by the file `mic_settings.sh` with the command
```
source $MICMAT_PATH/mic_settings.sh
```
where `$MICMAT_PATH` contains the path to the package. It may be convenient to add this command to the bash profile. 

In order to import the package into Python, simply add the command
```
import micmat_wrap as mm
```
to the top of your pure Python file.


# Usage



# License (BSD 3-Clause)
Copyright (c) 2014, Oren Rippel and Ryan P. Adams

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.