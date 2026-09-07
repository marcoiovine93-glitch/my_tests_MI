MODULE laxlib_cuda_stream_mod
#if def(__CUDA)
use cudafor
IMPLICIT NONE 
INTEGER(cuda_stream_kind) :: laxlib_cuda_stream !!! M.Iovine - laxlib_cuda_stream is a variable like a c++ static, it is available publicly and at the same time similar to an instance variable (related to this specific module)
INTEGER :: cusolver_thread

CONTAINS
SUBROUTINE initialize_laxlib_cuda_stream( stream, mypippo)
        IMPLICIT NONE
        INTEGER(cuda_stream_kind), INTENT(IN) :: stream
        INTEGER, INTENT(IN) :: mypippo
        cusolver_thread = mypippo
        laxlib_cuda_stream = stream
        print '("In initialize_laxlib_cuda_stream, thread ",I5,I24)', cusolver_thread, laxlib_cuda_stream
END SUBROUTINE initialize_laxlib_cuda_stream


#endif
END MODULE laxlib_cuda_stream_mod
