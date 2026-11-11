#!/bin/bash

for i in {4..9}; do
    echo "Running script for report files for david = $i"
    cd nk_11
    sqlite3 -header -column profile_cb_davidson_new_nk11_d${i}.sqlite "
    WITH KernelSummary AS (SELECT 
        s.value AS KernelName,
        COUNT(*) AS KernelCallCount,
        SUM(k.end - k.start) AS TotalDurationNs,
        AVG(k.end - k.start) AS AvgDurationNs,
        MIN(k.end - k.start) AS MinDurationNs,
        MAX(k.end - k.start) AS MaxDurationNs
    FROM 
        CUPTI_ACTIVITY_KIND_KERNEL AS k
    JOIN 
        StringIds AS s ON k.demangledName = s.id
    GROUP BY 
        KernelName
    )
    SELECT
        KernelName,
        KernelCallCount,
        TotalDurationNs,
        AvgDurationNs,
        -- Calculate percentage of total GPU active time consumed by this kernel
        (CAST(TotalDurationNs AS REAL) / (SELECT SUM(TotalDurationNs) FROM KernelSummary)) * 100 AS GpuTimeSharePercentage
    FROM
        KernelSummary
    ORDER BY
        TotalDurationNs DESC;
    " > ../output_results_dgx_OLD/output_dgxold_cb_davidson_new_nk11_d${i}.txt

    cd ../

    echo "Success for $i"
done
