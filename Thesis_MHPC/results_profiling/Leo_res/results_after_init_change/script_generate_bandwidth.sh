#!/bin/bash

for i in {4..9}; do
    echo "Running script for report files for david = $i"
    cd nk_4
    sqlite3 -header -column profile_cb_davidson_new_nk4_d${i}.sqlite "
    SELECT 
        streamId,
        copyKind,
        COUNT(*) AS TransferCount,
        SUM(bytes) / (1024.0 * 1024 * 1024) AS TotalSizeGB,
        (MAX(end) - MIN(start)) AS TimelineSpanNs,
        (CAST(SUM(bytes) AS REAL) / (MAX(end) - MIN(start))) AS EffectiveBandwidthGBps
    FROM 
        CUPTI_ACTIVITY_KIND_MEMCPY
    GROUP BY 
        streamId, copyKind;
    " > ../output_leo_NEW/output_leo_new_bandwidth_nk4_d${i}.txt

    cd ../
    

    echo "Running script for report files for david = $i"
    cd nk_11
    sqlite3 -header -column profile_cb_davidson_new_nk11_d${i}.sqlite "
    SELECT
         streamId,
         copyKind,
         COUNT(*) AS TransferCount,
         SUM(bytes) / (1024.0 * 1024 * 1024) AS TotalSizeGB,
         (MAX(end) - MIN(start)) AS TimelineSpanNs,
         (CAST(SUM(bytes) AS REAL) / (MAX(end) - MIN(start))) AS EffectiveBandwidthGBps
    FROM
         CUPTI_ACTIVITY_KIND_MEMCPY
    GROUP BY
         streamId, copyKind;
     " > ../output_leo_NEW/output_leo_new_bandwidth_nk11_d${i}.txt

    cd ../

    echo "Success for $i"
done
