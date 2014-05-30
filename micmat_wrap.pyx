cimport cython
from libc.stdlib cimport malloc, free
cimport cmicmat
import numpy as np
cimport numpy as np
import time

cdef class RandGen:
    cdef public int skip_num

    def __cinit__(self):
        self.skip_num = 0


cdef class MICMat:
    #cdef int ROWS, COLS
    cdef tuple shape
    cdef int offloaded
    cdef int persist # if is set to 1, memory is NOT deallocated by garbage collector. Use with caution!
    cdef int dtype # 0 is float, 1 is int
    cdef float *A
    cdef int *A_int

    def __cinit__(self, *args):
        if not args:
            pass

        else:
            if type(args[0]) == tuple:
                shape = args[0]
                self.allocate_host(shape)

            elif type(args[0]) == np.ndarray:
                B = args[0]
                
                assert B.dtype == np.float32 or B.dtype == np.float64, 'Unrecognized data type given to MICMat: ' + `B.dtype` + '.'
                assert B.ndim == 2, 'Array given to MICMat has wrong number of dimensions: ' + B.ndim + '.'
                
                self.copy_host(B)

            elif type(args[0]) == MICMat:
                B = args[0]
                self.copy_host(B)

            else:
                raise NameError('Unrecognized type given to MICMat: ' + `type(args[0])` + '.')


######### allocation and deallocation functions
    def allocate_host(self, *args):
        if not args:
            assert self.ROWS * self.COLS > 0, 'MICMat shape not set. Cannot allocate.'
            self.A = cmicmat.allocate_host(self.size)

        else:
            shape = args[0]
            self.shape = shape
            self.A = cmicmat.allocate_host(self.size)

        self.dtype = 0
        return self

    def allocate(self, *args):
        if not args:
            self.allocate_host()

        else:
            self.allocate_host(args[0])

        self.offload_mic()
        return self

    def free_mic_float(self):
        cmicmat.free_mic(self.size, self.A)
        self.offloaded = False
        
        return self

    def free_mic_int(self):
        cmicmat.free_mic_int(self.size, self.A_int)
        self.offloaded = False
        
        return self

    def free_host_float(self):
        cmicmat.free_host(self.size, self.A)
        self.shape = (0, 0)
        return self

    def free_host_int(self):
        cmicmat.free_host_int(self.size, self.A_int)
        self.shape = (0, 0)
        return self

    def free_mic(self):
        if self.dtype == 0:
            self.free_mic_float()

        elif self.dtype == 1:
            self.free_mic_int()

        self.offloaded = False

    def free_float(self):
        if self.offloaded: 
            self.free_mic_float()

        self.free_host_float()

        return self

    def free_int(self):
        if self.offloaded: 
            self.free_mic_int()
        
        self.free_host_int()
        return self

    def free(self):
        if self.dtype == 0:
            self.free_float()

        elif self.dtype == 1:
            self.free_int()

    def __dealloc__(self):
        if self.size > 0 and not self.persist: 
            self.free()


######### offloading and pulling functions
    def offload_mic(self):
        if not self.offloaded:
            cmicmat.offload_mic(self.size, self.A)
            self.offloaded = True
        else:
            cmicmat.push_mic(self.size, self.A)    
        
        return self

    def offload_host(self):
        if self.offloaded:
            self.pull_mic()

        self.free_mic()
        return self

    #def push_mic(self):
    #    cmicmat.pull_mic(self.size, self.A)
    #    return self

    def pull_mic(self):
        cmicmat.pull_mic(self.size, self.A)
        return self


######### attribute getting, setting, casting
    def __getattr__(self, name):
        if name == 'shape':
            #return [self.ROWS, self.COLS]
            return self.shape

        if name == 'ROWS':
            assert len(self.shape) == 2, 'MICMat shape is ' + `self.shape` + ', so cannot request for ROWS attribute.'
            return self.shape[0]

        if name == 'COLS':
            assert len(self.shape) == 2, 'MICMat shape is ' + `self.shape` + ', so cannot request for COLS attribute.'
            return self.shape[1]

        elif name == 'size':
            #return self.ROWS*self.COLS
            return np.prod(self.shape)

        elif name == 'dtype':
            return self.dtype

        elif name == 'offloaded':
            return self.offloaded

        else:
            raise NameError('Cannot get attribute \'' + name + '\'.')

    def __setattr__(self, name, value):
        if name == 'shape':
            # in pure Python need command of the form object.__setattr__(self, 'ROWS', value[0])
            # but here we're using extension types, so direct assignment is used
            self.shape = value
            #self.ROWS = value[0]
            #self.COLS = value[1]

        #if name == 'ROWS':
        #    assert len(self.shape) == 2, 'MICMat shape is ' + `self.shape` + ', so cannot set ROWS attribute.'
        #    self.shape = (value, self.shape[1])

        #if name == 'COLS':
        #    assert len(self.shape) == 2, 'MICMat shape is ' + `self.shape` + ', so cannot set COLS attribute.'
        #    self.shape = (self.shape[0], value)

        elif name == 'offloaded':
            self.offloaded = value

        elif name == 'dtype':
            self.dtype = value

        else:
            raise NameError('Cannot set attribute \'' + name + '\'.')

    def astype(self, dtype):
        # we assume that there are only 2 data types... float and int
        if dtype == 'float':
            self.A = cmicmat.cast_float(self.size, self.A_int, self.offloaded)
            self.dtype = 0

        elif dtype == 'int':
            assert 1 == 2, 'Haven\'t implemented int casting yet.'
            self.dtype = 1

        return self


######### copy and replacement
    def copy_host(self, B):
        if self.size > 0: 
            self.free()
            
        cdef MICMat B_MICMat
        cdef np.ndarray[np.float32_t, ndim = 1] B_float

        if type(B) == MICMat:
            B_MICMat = B
            self.allocate_host(B_MICMat.shape)
            cmicmat.copy(B_MICMat.size, self.A, B_MICMat.A, 0)
        
        elif type(B) == np.ndarray:
            self.allocate_host(B.shape)  
            B_float = B.ravel(order = 'C').astype(np.float32)
            cmicmat.copy(B_float.size, self.A, <float *> B_float.data, 0)

        return self

    def copy_mic(self, MICMat B):
        assert B.offloaded, 'MICMat object copied on MIC is not, in fact, on the MIC.'

        if self.offloaded: 
            self.free_mic()

        cmicmat.copy(B.size, self.A, B.A, 1)
        self.offloaded = True

        return self

    def deepcopy(self):
        cdef MICMat B = MICMat()

        if self.offloaded: 
            B.allocate(self.shape)
            cmicmat.deepcopy(self.size, B.A, self.A)
        
        else:
            B.copy_host(self)

        return B

    def replace_host(self, B):
        cdef MICMat B_MICMat
        cdef np.ndarray[np.float32_t, ndim = 1] B_float

        if type(B) == MICMat:
            B_MICMat = B
            assert self.shape == B_MICMat.shape, 'Matrix dimensions ' + `self.shape` + ' and ' + `B_MICMat.shape` + ' don\'t match in update.'
            cmicmat.replace_host(self.size, self.A, B_MICMat.A)

        elif type(B) == np.ndarray:
            assert self.shape == B.shape, 'Matrix dimensions ' + `self.shape` + ' and ' + `B.shape` + ' don\'t match in update.'    
            B_float = B.ravel(order = 'C')
            cmicmat.replace_host(self.size, self.A, <float *> B_float.data)

        return self

    def replace_mic(self, B):
        cdef MICMat B_MICMat
        cdef np.ndarray[np.float32_t, ndim = 1] B_float

        if type(B) == MICMat:
            B_MICMat = B
            assert self.shape == B_MICMat.shape, 'Matrix dimensions ' + `self.shape` + ' and ' + `B_MICMat.shape` + ' don\'t match in update.'
            cmicmat.replace_mic(self.size, self.A, B_MICMat.A)

        elif type(B) == np.ndarray:
            assert self.shape == B.shape, 'Matrix dimensions ' + `self.shape` + ' and ' + `B.shape` + ' don\'t match in update.'
            B_float = B.ravel(order = 'C')
            cmicmat.replace_mic(self.size, self.A, <float *> B_float.data)

        return self

    def replace(self, B):
        self.replace_host(B)

        if self.offloaded:
            self.replace_mic(B)

        return self


######### slicing
    def slice_inds(self, indices):
        cdef np.ndarray[np.int32_t, ndim = 1] indices_np
        cdef MICMat indices_MICMat, A_sliced

        if type(indices) == MICMat:
            A_sliced = MICMat()
            A_sliced.shape = indices.shape
            indices_MICMat = indices
            indices_MICMat.dtype = 1
            A_sliced.A = cmicmat.slice_inds(indices_MICMat.size, indices_MICMat.A_int, self.A, indices_MICMat.offloaded, self.offloaded)

        elif type(indices) == list or type(indices) == np.ndarray:
            A_sliced = MICMat()
            A_sliced.shape = (1, len(indices))
            indices_np = np.array(indices, dtype = np.int32)
            assert indices_np.max() < self.size and indices_np.min() >= 0, 'Indices specified are out of bounds for array of size ' + `self.size` + '.'
            A_sliced.A = cmicmat.slice_inds(len(indices), <int *> indices_np.data, self.A, 0, self.offloaded)

        A_sliced.offloaded = self.offloaded
        return A_sliced

    def __getslice__(self, i, j):
        return self.slice_inds(range(i, j)) 

    def slice_cols(self, indices):     
        cdef np.ndarray[np.int32_t, ndim = 1] indices_np
        cdef MICMat indices_MICMat, A_sliced

        if type(indices) == MICMat:
            A_sliced = MICMat()
            A_sliced.shape = (self.ROWS, indices.size)
            indices_MICMat = indices
            indices_MICMat.dtype = 1
            A_sliced.A = cmicmat.slice_cols(indices_MICMat.size, indices_MICMat.A_int, self.ROWS, self.COLS, self.A, indices_MICMat.offloaded, self.offloaded)

        elif type(indices) == list or type(indices) == np.ndarray:
            A_sliced = MICMat()
            A_sliced.shape = (self.ROWS, len(indices))
            indices_np = np.array(indices, dtype = np.int32)
            assert indices_np.max() < self.COLS and indices_np.min() >= 0, 'Indices specified are out of bounds for array of shape ' + `self.shape` + '.'
            A_sliced.A = cmicmat.slice_cols(len(indices), <int *> indices_np.data, self.ROWS, self.COLS, self.A, 0, self.offloaded)

        A_sliced.offloaded = self.offloaded
        return A_sliced

    def slice_rows(self, indices):     
        cdef np.ndarray[np.int32_t, ndim = 1] indices_np
        cdef MICMat indices_MICMat, A_sliced

        if type(indices) == MICMat:
            A_sliced = MICMat()
            A_sliced.shape = (indices.size, self.COLS)
            indices_MICMat = indices
            indices_MICMat.dtype = 1
            A_sliced.A = cmicmat.slice_rows(indices_MICMat.size, indices_MICMat.A_int, self.ROWS, self.COLS, self.A, indices_MICMat.offloaded, self.offloaded)

        elif type(indices) == list or type(indices) == np.ndarray:
            A_sliced = MICMat()
            A_sliced.shape = (len(indices), self.COLS)
            indices_np = np.array(indices, dtype = np.int32)

            assert indices_np.max() < self.ROWS and indices_np.min() >= 0, 'Indices specified are out of bounds for array of shape ' + `self.shape` + '.'
            A_sliced.A = cmicmat.slice_rows(len(indices), <int *> indices_np.data, self.ROWS, self.COLS, self.A, 0, self.offloaded)

        A_sliced.offloaded = self.offloaded
        return A_sliced


######### random (and non-random) number generation
    def fill_randn(self, RandGen stream, float mu, float sigma):
        cmicmat.fill_randn(stream.skip_num, self.size, self.A, mu, sigma)
        stream.skip_num = stream.skip_num + self.size
        return self

    def fill_uniform(self, RandGen stream):
        cmicmat.fill_uniform(stream.skip_num, self.size, self.A)
        stream.skip_num = stream.skip_num + self.size
        return self

    def fill_bernoulli(self, RandGen stream, float p):
        cmicmat.fill_bernoulli(stream.skip_num, self.size, self.A, p)
        stream.skip_num = stream.skip_num + self.size
        return self

    def fill_zeros(self):
        cmicmat.fill_zeros(self.size, self.A, self.offloaded)
        return self  

    def fill_ones(self):
        cmicmat.fill_ones(self.size, self.A, self.offloaded)
        return self


######### string and array outputting
    def print_host(self):
        cmicmat.print_slice(self.shape[-2], self.shape[-1], self.A, 0)

    def print_mic(self):
        cmicmat.print_slice(self.shape[-2], self.shape[-1], self.A, 1)

    def __str__(self):
        if self.offloaded:
            print 'MICMat of shape ' + `self.shape` + ' on MIC:'
            self.print_mic()
        else:
            print 'MICMat of shape ' + `self.shape` + ' on host:'
            self.print_host()

        return ''

    def __repr__(self):
        description_str = ''
        if self.ROWS == 1 and self.COLS == 1:
            description_str = `self.float()`

        return description_str

    def array(self):
        cdef np.ndarray[np.float32_t, ndim = 2] S = np.zeros([self.ROWS, self.COLS], dtype = np.float32)

        if self.offloaded:
            self.pull_mic()
        
        S.data = <char *> cmicmat.unalign_host(self.size, self.A)
        
        return S

    def ndarray(self):
        cdef np.ndarray[np.float32_t, ndim = 4] S = np.zeros(self.shape, dtype = np.float32)

        if self.offloaded:
            self.pull_mic()
        
        S.data = <char *> cmicmat.unalign_host(self.size, self.A)
        
        return S

    def float(self):
        assert self.size == 1, 'MICMat size must be 1 to convert to float.'

        if self.offloaded:
            self.pull_mic()
        
        cdef np.float32_t S = cmicmat.output_float(self.A)
        
        return S


######### elementwise operations
    def exp(self):
        cmicmat.expo(self.size, self.A)
        return self

    def floor(self):
        cmicmat.flooro(self.size, self.A)
        return self

    def sign(self):
        cmicmat.sign(self.size, self.A)
        return self
 
    def log(self):
        cmicmat.lg(self.size, self.A)
        return self

    def abs(self):
        cmicmat.abso(self.size, self.A)
        return self

    def sqrt(self):
        cmicmat.sqrto(self.size, self.A) 
        return self

    def scale(self, float c):
        cmicmat.scale(self.size, self.A, c)
        return self

    def pow(self, b):
        cmicmat.powo(self.size, self.A, <float> b) 
        return self

    def invert(self):
        cmicmat.invert(self.size, self.A)
        return self

    def clip(self, float lower, float upper):
        cmicmat.clip(self.size, self.A, lower, upper)
        return self

    def clip_low(self, float lower):
        cmicmat.clip_low(self.size, self.A, lower)
        return self


######### matrix operations
    def convolve_replace(self, MICMat inputs, MICMat filters, int stride, style):
        # asserts that check number of dimensions, sizes, etc
        N, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        K, Y, X = filters.shape[0], filters.shape[2], filters.shape[3]

        cdef int tight
        # output shape must take into account stride hyperparameters!
        if style == 'full':
            assert self.shape == (N, K, H + Y - 1, W + X - 1), 'Output shape is ' + self.shape + ' rather than ' + `(N, K, H + Y - 1, W + X - 1)` + '.'
            tight = 0

        elif style == 'tight':
            assert self.shape == (N, K, H - Y + 1, W - X + 1), 'Output shape is ' + self.shape + ' rather than ' + `(N, K, H - Y + 1, W - X + 1)` + '.'
            tight = 1

        else:
            raise NameError('Convolution style not recognized.')
        
        cmicmat.convolve(N, C, H, W, inputs.A, K, Y, X, filters.A, self.A, tight)

    def T(self):
        cdef MICMat S = self.deepcopy()

        cmicmat.T(S.ROWS, S.COLS, S.A)
        S.shape = (self.shape[1], self.shape[0])

        return S

    def dot(self, MICMat V, *args):
        cdef MICMat S
        cdef int T_A = False
        cdef int T_B = False
        cdef COLS_LEFT, ROWS_RIGHT, ROWS_LEFT, COLS_RIGHT

        if not args:
            pass

        else:
            T_A = args[0]
            T_B = args[1]

        if not T_A:
            COLS_LEFT = self.COLS
            ROWS_LEFT = self.ROWS
        else: 
            COLS_LEFT = self.ROWS
            ROWS_LEFT = self.COLS

        if not T_B:
            ROWS_RIGHT = V.ROWS
            COLS_RIGHT = V.COLS
        else: 
            ROWS_RIGHT = V.COLS
            COLS_RIGHT = V.ROWS
        
        assert COLS_LEFT == ROWS_RIGHT, 'Matrix dimensions ' + `self.shape` + ' and ' + `V.shape` + ' don\'t match in matrix product.'
        
        S = MICMat([ROWS_LEFT, COLS_RIGHT])
        if self.offloaded:
            S.offload_mic()
            
        cmicmat.dot_replace(self.ROWS, self.COLS, T_A, self.A, V.ROWS, V.COLS, T_B, V.A, 0., S.A)

        return S

    def dot_replace(self, MICMat M, MICMat V, *args):
    # self = M*V
        cdef int T_A = False
        cdef int T_B = False
        cdef COLS_LEFT, ROWS_RIGHT, ROWS_LEFT, COLS_RIGHT

        if not args:
            pass

        else:
            T_A = args[0]
            T_B = args[1]

        if not T_A:
            COLS_LEFT = M.COLS
            ROWS_LEFT = M.ROWS
        else: 
            COLS_LEFT = M.ROWS
            ROWS_LEFT = M.COLS

        if not T_B:
            ROWS_RIGHT = V.ROWS
            COLS_RIGHT = V.COLS
        else: 
            ROWS_RIGHT = V.COLS
            COLS_RIGHT = V.ROWS


        assert COLS_LEFT == ROWS_RIGHT, 'Matrix dimensions ' + `M.shape` + ' and ' + `V.shape` + ' don\'t match in matrix product.'
        assert self.ROWS == ROWS_LEFT and self.COLS == COLS_RIGHT, 'Matrix product of matrices of dimensions ' + `M.shape` + ' and ' + `V.shape` + ' doesn\'t match output MICMat ' + `self.shape` + '.'
        
        cmicmat.dot_replace(M.ROWS, M.COLS, T_A, M.A, V.ROWS, V.COLS, T_B, V.A, 0., self.A)

        return self

    def dot_replace_update(self, MICMat M, MICMat V, *args):
    # self = self + M*V
        cdef int T_A = False
        cdef int T_B = False
        cdef COLS_LEFT, ROWS_RIGHT, ROWS_LEFT, COLS_RIGHT

        if not args:
            pass

        else:
            T_A = args[0]
            T_B = args[1]

        if not T_A:
            COLS_LEFT = M.COLS
            ROWS_LEFT = M.ROWS
        else: 
            COLS_LEFT = M.ROWS
            ROWS_LEFT = M.COLS

        if not T_B:
            ROWS_RIGHT = V.ROWS
            COLS_RIGHT = V.COLS
        else: 
            ROWS_RIGHT = V.COLS
            COLS_RIGHT = V.ROWS


        assert COLS_LEFT == ROWS_RIGHT, 'Matrix dimensions ' + `M.shape` + ' and ' + `V.shape` + ' don\'t match in matrix product.'
        assert self.ROWS == ROWS_LEFT and self.COLS == COLS_RIGHT, 'Matrix product of matrices of dimensions ' + `M.shape` + ' and ' + `V.shape` + ' doesn\'t match output MICMat ' + `self.shape` + '.'
        
        cmicmat.dot_replace(M.ROWS, M.COLS, T_A, M.A, V.ROWS, V.COLS, T_B, V.A, 1., self.A)

    def dot_vec(self, MICMat B):
    # sum(self .* B)
        cdef MICMat S = MICMat()
        S.shape = (1, 1)
        assert self.size == B.size, 'Matrix dimensions don\'t match for dot vector product.'
        S.A = cmicmat.dot_vec(self.size, self.A, B.A)
        S.offloaded = self.offloaded

        return S 

    def update(self, V, *args):
        cdef float ALPHA
        cdef MICMat V_MICMat
        if not args:
            ALPHA = 1.0
        else:
            ALPHA = args[0]

        if type(V) == MICMat:
            V_MICMat = V
            assert (self.ROWS == V_MICMat.ROWS and self.COLS == V_MICMat.COLS) or (self.ROWS == V_MICMat.ROWS and V_MICMat.COLS == 1) or (V_MICMat.ROWS == 1 and self.COLS == V_MICMat.COLS) or (V_MICMat.ROWS == 1 and V_MICMat.COLS == 1), 'Matrix dimensions ' + `self.shape` + ' and ' + `V.shape` + ' don\'t match in update.'
            cmicmat.update(self.ROWS, self.COLS, self.A, V_MICMat.ROWS, V_MICMat.COLS, V_MICMat.A, ALPHA)

        elif type(V) == np.float32 or type(V) == np.float64 or type(V) == float:
            if type(V) == np.float64:
                V = V.astype(np.float32)
            cmicmat.update_const(self.size, self.A, ALPHA*V)

        return self

    def multiply(self, V):
        cdef MICMat V_MICMat

        if type(V) == MICMat:
            V_MICMat = V
            assert (self.ROWS == V_MICMat.ROWS and self.COLS == V_MICMat.COLS) or (self.ROWS == V_MICMat.ROWS and V_MICMat.COLS == 1) or (V_MICMat.ROWS == 1 and self.COLS == V_MICMat.COLS) or (V_MICMat.ROWS == 1 and V_MICMat.COLS == 1), 'Matrix dimensions ' + `self.shape` + ' and ' + `V.shape` + ' don\'t match in update.'
            cmicmat.mult(self.ROWS, self.COLS, self.A, V_MICMat.ROWS, V_MICMat.COLS, V_MICMat.A)
        elif type(V) == np.float32 or type(V) == np.float64 or type(V) == float or type(V) == int:
            if type(V) == np.float64:
                V = V.astype(np.float32)
            cmicmat.scale(self.size, self.A, V)

        return self

    def divide(self, V):
        cdef MICMat V_MICMat

        if type(V) == MICMat:
            V_MICMat = V
            assert (self.ROWS == V_MICMat.ROWS and self.COLS == V_MICMat.COLS) or (self.ROWS == V_MICMat.ROWS and V_MICMat.COLS == 1) or (V_MICMat.ROWS == 1 and self.COLS == V_MICMat.COLS) or (V_MICMat.ROWS == 1 and V_MICMat.COLS == 1), 'Matrix dimensions ' + `self.shape` + ' and ' + `V.shape` + ' don\'t match in update.'
            cmicmat.divide(self.ROWS, self.COLS, self.A, V_MICMat.ROWS, V_MICMat.COLS, V_MICMat.A)
        elif type(V) == np.float32 or type(V) == np.float64 or type(V) == float or type(V) == int:
            if type(V) == np.float64:
                V = V.astype(np.float32)
            cmicmat.scale(self.size, self.A, 1.0/V)

        return self

    def sum(self, *args):         
        cdef MICMat S = MICMat()
        cdef int AXIS = 2

        if not args or args[0] == 2:
            S.shape = (1, 1)
            S.A = cmicmat.sum_axis(self.size, 1, self.A, AXIS)
        else:
            AXIS = args[0]
            if AXIS == 0:
                S.shape = (1, self.COLS)

            if AXIS == 1:
                S.shape = (self.ROWS, 1)
            
            S.A = cmicmat.sum_axis(self.ROWS, self.COLS, self.A, AXIS)    
        
        S.offloaded = self.offloaded
        return S 

    def max_and_arg(self, *args):         
        cdef MICMat I = MICMat()
        cdef int AXIS = 2

        if not args or args[0] == 2:
            I.shape = (1, 1)

        else:
            AXIS = args[0]
            if AXIS == 0:
                I.shape = (1, self.COLS)

            if AXIS == 1:
                I.shape = (self.ROWS, 1)
        
        I.A_int = cmicmat.max_axis(self.ROWS, self.COLS, self.A, AXIS)
        I.dtype = 1
        I.offloaded = self.offloaded
        cdef MICMat S = self.slice_inds(I)
        
        if AXIS != 2:
            cmicmat.index_global_to_local(self.ROWS, self.COLS, I.A_int, AXIS)
        
        return S, I

    def max(self, *args):
        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        S, _ = self.max_and_arg(AXIS)
        return S

    def min(self, *args):
        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        return self.scale(-1.).max(AXIS).scale(-1.)

    def argmax(self, *args):
        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        _, I = self.max_and_arg(AXIS)
        return I

    def mean(self, *args):
        cdef int AXIS = 2 
        cdef float SCALING

        if not args or args[0] == 2:
            SCALING = 1.0/(<float> self.size)
        else:
            AXIS = args[0]
            if AXIS == 0:
                SCALING = 1.0/(<float> self.ROWS)

            if AXIS == 1:
                SCALING = 1.0/(<float> self.COLS)

        
        cdef MICMat S = self.sum(AXIS)
        S.offloaded = self.offloaded
        S.scale(SCALING)
        return S      

    def norm(self, *args):

        cdef int AXIS = 2 
        if not args:
            pass
        else:
            AXIS = args[0]
        
        cdef MICMat B = self.deepcopy()
        B.pow(2.0)

        cdef MICMat S = B.sum(AXIS)
        S.offloaded = self.offloaded
        S.sqrt()
        B.free()
        return S 

    def var(self, *args):         
        cdef MICMat B = self.deepcopy()
        B.pow(2.0)

        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        cdef MICMat D = self.mean(AXIS)
        D.pow(2.0)
        
        B.update(D, -1.0)   
        
        cdef MICMat S = B.mean(AXIS)
        S.offloaded = self.offloaded

        return S

    def std(self, *args):
        cdef int AXIS = 2
        if not args:
            pass
        else:
            AXIS = args[0]

        cdef MICMat S = self.var(AXIS)
        S.offloaded = self.offloaded
        S.sqrt()
        return S


######### native Python operator implementations
    def __add__(self, V):
        cdef MICMat S
        if type(self) == float:
            S = V.deepcopy()
            S.update(self)
        else:
            S = self.deepcopy()
            S.update(V)

        return S

    def __sub__(self, V):
        cdef MICMat S
        if type(self) == float:
            S = -V.deepcopy()
            S.update(self)
        else:
            S = self.deepcopy()
            S.update(V, -1.0)

        return S
        
    def __mul__(self, V):
        cdef MICMat S
        if type(self) == float:
            S = V.deepcopy()
            S.multiply(self)
        else:
            S = self.deepcopy()
            S.multiply(V)

        return S

    def __div__(self, V):
        cdef MICMat S
        if type(self) == float:
            S = V.deepcopy()
            S.invert()
            return self * S
        else:
            S = self.deepcopy()
            S.divide(V)
        
        return S

    def __abs__(self):
        cdef MICMat S = self.deepcopy()
        return S.abs()

    def __neg__(self):
        cdef MICMat S = self.deepcopy()
        return S * (-1.0)
        

    def __iadd__(self, V):
        self.update(V)
        return self

    def __isub__(self, V):
        self.update(V, -1.0)
        return self

    def __imul__(self, V):
        self.multiply(V)
        return self

    def __idiv__(self, V):
        self.divide(V)
        return self

    def __len__(self):
        return self.size

    def equal(self, MICMat V):
        assert self.shape == V.shape, 'The shapes ' + `self.shape` + ' and ' + `V.shape` + ' of the compared MICMats don\'t match.'
        cdef MICMat S = MICMat()
        S.shape = V.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.equal(self.size, self.A, V.A)

        return S

    def leq(self, float V):
        cdef MICMat S = MICMat()
        S.shape = self.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.leq(self.size, self.A, V)

        return S

    def geq(self, float V):
        cdef MICMat S = MICMat()
        S.shape = self.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.geq(self.size, self.A, V)

        return S

    def greater(self, float V):
        cdef MICMat S = MICMat()
        S.shape = self.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.greater(self.size, self.A, V)

        return S

    def __or__(MICMat self, MICMat V):
        assert self.shape == V.shape, 'The shapes ' + `self.shape` + ' and ' + `V.shape` + ' of the compared MICMats don\'t match.'
        cdef MICMat S = MICMat()
        S.shape = V.shape
        S.offloaded = self.offloaded
        S.A = cmicmat.elementwise_or(self.size, self.A, V.A)
        return S

    def __richcmp__(self, V, int operation):
        if operation == 1:
            assert type(V) == float or type(V) == np.float32, 'Currently can only do comparison for RHS being a float.'
            return self.leq(V)

        elif operation == 2:
            return self.equal(V)

        elif operation == 3:
            return self.equal(V).update(-1.).scale(-1.)

        elif operation == 4:
            assert type(V) == float or type(V) == np.float32, 'Currently can only do comparison for RHS being a float.'
            return self.greater(V)

        elif operation == 5:
            assert type(V) == float or type(V) == np.float32, 'Currently can only do comparison for RHS being a float.'
            return self.geq(V)


######### misc
    def labels_to_vectors(self, int K):
        cdef MICMat S = MICMat()
        S.shape = (self.shape[0], K)
        S.offloaded = self.offloaded
        S.A = cmicmat.labels_to_vectors(self.shape[0], K, self.A)

        return S   

    def frac_zero(self):
        cdef MICMat zeros = MICMat(self.shape) 
        if self.offloaded:
            zeros.offload_mic()

        zeros.fill_zeros()
        return (self == zeros).sum().float()/self.size


cdef class Scratch(MICMat):
    cdef list shapes

    def __cinit__(self, *args):
        self.shapes = []

    def __getattr__(self, name):
        if name == 'shapes':
            return self.shapes

        else:
            return MICMat.__getattr__(self, name)

    def reset(self, MICMat V):
        self.shapes = []
        return self.append(V)

    def append(self, MICMat V):
        assert self.size >= self.contents_size() + V.size, 'The scratch space is out of memory!'

        cmicmat.replace_partial_host(self.size, self.contents_size(), self.A, V.size, 0, V.A)

        if self.offloaded:
            cmicmat.replace_partial_mic(self.size, self.contents_size(), self.A, V.size, 0, V.A)

        self.shapes.append(V.shape)

        return self.get(len(self.shapes) - 1)

    def get(self, index):
        cdef MICMat S = MICMat()
        S.shape = self.shapes[index]
        S.offloaded = self.offloaded
        S.persist = 1
        S.A = cmicmat.get_partial(S.size, self.contents_size(index), self.A)

        return S

    def contents_size(self, *args):
        if not args:
            return sum([shape[0]*shape[1] for shape in self.shapes])
        
        else:
            index = args[0]
            return sum([shape[0]*shape[1] for shape in self.shapes[:index]])

def ping_each_core():
        print 'Pinging each core on the MIC.'
        cmicmat.ping_each_core()

def check_mic_status():
    cmicmat.check_mic_status()

def tester():
    cmicmat.tester()