# Copyright (c) 2014, Oren Rippel and Ryan P. Adams
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


rm micmat.o
rm micmat_wrap.c
rm micmat_wrap.so
rm libmicmat.so

########### Build library
source mic_settings.sh

echo -e "Compiling C library. \n\n"

 icc -c micmat.c -o micmat.o -V -O3 -std=c99 -openmp \
 -fpic -mkl

echo -e "Linking C library. \n\n"

icc -V -shared -o libmicmat.so micmat.o -lc \
 -fpic -lm -mkl

echo -e "Building Cython wrapper. \n\n"

############ Cython wrapping
CC="icc"   \
CXX="icc"   \
CFLAGS="-I$MICMAT_PATH"   \
LDFLAGS="-L$MICMAT_PATH"   \
    python micmat_setup.py build_ext -i