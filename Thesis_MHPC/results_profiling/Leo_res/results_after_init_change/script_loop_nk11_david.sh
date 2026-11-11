#!/bin/bash

for i in {4..9}; do
    echo "Running script for report files for david = $i"

    cd build/bin || exit 1

    nsys profile --trace=cuda,openacc,openmp,nvtx \
    --stats=true \
    --force-overwrite=true \
    -o profile_cb_davidson_new_${i} \
    ./cb_davidson.x < ../../examples/si2_11_points_d${i}.in

    cp profile* ../../results
    
    cd ../../

    echo "Success for $i"
done
