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