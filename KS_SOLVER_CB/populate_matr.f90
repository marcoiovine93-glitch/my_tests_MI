SUBROUTINE populate_array(matr, n, general_matr)
     USE omp_lib, ONLY: omp_get_thread_num
     INTEGER :: i_th
     
     matr(1,1) = COMPLEX(2.0D0, 0.0D0, kind=8)
     matr(2,2) = COMPLEX(2.0D0, 0.0D0, kind=8)
     matr(3,3) = COMPLEX(2.0D0, 0.0D0, kind=8)
     matr(1,2) = COMPLEX(0.0D0, 0.0D0, kind=8)
     matr(1,3) = COMPLEX(0.0D0, -1.0D0, kind=8)
     matr(2,3) = COMPLEX(0.0D0, 0.0D0, kind=8)
     matr(3,1) = COMPLEX(0.0D0, 1.0D0, kind=8)
     matr(3,2) = COMPLEX(0.0D0, 0.0D0, kind=8)
     matr(2,1) = COMPLEX(0.0D0, 0.0D0, kind=8)
     
     #if defined(__OMP)
        i_th = omp_get_thread_num
     #endif

     general_matr(:,:,i_th) = matr
     
     !! WE NEED TO BE SURE THAT ALL THE THREADS POPULATE THE 3D ARRAY WITH THE CORRESPONDING SLICE OF DATA!!!
     !$omp barrier 
     
     !! We copy the 3d array on the device before the call to the diaghg or inside the diaggh?????

     !!! Only a single thread perfrms the call to the subroutine where it is defined the batched kernel call!!!
     if (i_th == 0)
         call diaghg_new_batch()
     endif




