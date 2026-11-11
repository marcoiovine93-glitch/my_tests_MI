#!/bin/bash

for i in {4..9}; do
    echo "Running script for report files for david = $i"
    cd nk_11
    sqlite3 -header -column profile_cb_davidson_nk_11_d${i}.sqlite "
SELECT 
    (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL) AS TotalKernelCalls,
    SUM(end - start) AS TotalKernelExecutionNs,
    (CAST(SUM(end - start) AS REAL) / (SELECT (MAX(end) - MIN(start)) FROM CUPTI_ACTIVITY_KIND_KERNEL)) * 100 AS OverallGpuActivePercentage
FROM 
    CUPTI_ACTIVITY_KIND_KERNEL;
" > ../output_results_dgx_NEW/total_gpu_usage_dgx_new_nk11_d${i}.txt    

    cd ../

    echo "Success for $i"
    
done

 for i in {4..9}; do
     echo "Running script for report files for david = $i"
     cd nk_4
     sqlite3 -header -column profile_cb_davidson_nk_4_d${i}.sqlite "
 SELECT
     (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL) AS TotalKernelCalls,
     SUM(end - start) AS TotalKernelExecutionNs,
     (CAST(SUM(end - start) AS REAL) / (SELECT (MAX(end) - MIN(start)) FROM CUPTI_ACTIVITY_KIND_KERNEL)) * 100 AS OverallGpuActivePercentage
 FROM
     CUPTI_ACTIVITY_KIND_KERNEL;
 " > ../output_results_dgx_NEW/total_gpu_usage_dgx_new_nk4_d${i}.txt

     cd ../

     echo "Success for $i"

 done

