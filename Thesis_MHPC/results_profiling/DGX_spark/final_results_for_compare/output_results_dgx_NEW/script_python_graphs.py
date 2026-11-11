import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# 1. Discover all summary/total files automatically
# Looking for files matching pattern like total_gpu_usage_*.txt
total_files = glob.glob("total_gpu_usage_dgx_new_nk4_d*.txt")

data = []

for filepath in total_files:
    filename = os.path.basename(filepath)
    
    # Extract parameter d from filename using regex (e.g., d9 -> 9, d11 -> 11)
    d_match = re.search(r'_d(\d+)', filename)
    if not d_match:
        continue
    d_val = int(d_match.group(1))
    
    # Determine version (Old vs New / Before vs After) from filename
    if 'old' in filename.lower():
        version = 'Before Changes (Old)'
    elif 'new' in filename.lower():
        version = 'After Changes (New)'
    else:
        version = 'Unknown'
        
    # Parse the total usage file
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 3:
                # Line 0: headers, Line 1: dashes, Line 2: values
                values = lines[2].split()
                total_calls = int(values[0])
                total_duration_ns = float(values[1])
                gpu_active_pct = float(values[2])
                
                data.append({
                    'd': d_val,
                    'Version': version,
                    'TotalKernelCalls': total_calls,
                    'TotalDurationSec': total_duration_ns / 1e9,  # Convert ns to seconds
                    'GpuActivePercentage': gpu_active_pct,
                    'Filename': filename
                })
    except Exception as e:
        print(f"Error parsing {filename}: {e}")

# Convert to DataFrame
df = pd.DataFrame(data)

if df.empty:
    print("No matching 'total_gpu_usage_*.txt' files found. Please check your filenames.")
else:
    # Sort by parameter d and version for clean plotting
    df = df.sort_values(by=['d', 'Version'])
    
    print("\n--- Extracted Scaling Summary Table ---")
    print(df[['d', 'Version', 'TotalKernelCalls', 'TotalDurationSec', 'GpuActivePercentage']].to_string(index=False))

    # 2. Generate Multi-Panel Comparison Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot A: Total Kernel Calls vs d
    for version, group in df.groupby('Version'):
        axes[0].plot(group['d'], group['TotalKernelCalls'], marker='o', markersize=8, linestyle='None', linewidth=2, label=version)
    axes[0].set_title('Total Kernel Invocations vs. David parameter', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Parameter d', fontsize=10)
    axes[0].set_ylabel('Total Kernel Call Count', fontsize=10)
    #axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot B: Total Execution Time vs d
    for version, group in df.groupby('Version'):
        axes[1].plot(group['d'], group['TotalDurationSec'], marker='s', markersize=8, linestyle='None', label=version)
    axes[1].set_title('Total Execution Time vs. David parameter', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Parameter d', fontsize=10)
    axes[1].set_ylabel('Total Duration (seconds)', fontsize=10)
    #axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Plot C: Overall GPU Active Percentage vs d
    for version, group in df.groupby('Version'):
        axes[2].plot(group['d'], group['GpuActivePercentage'], marker='^', markersize=8, linestyle='None', label=version)
    axes[2].set_title('GPU Active Percentage vs. David parameter', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Parameter d', fontsize=10)
    axes[2].set_ylabel('GPU Active Time (%)', fontsize=10)
    #axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    fig.suptitle('GPU data for nk_batches = 11 for the new code')

    plt.tight_layout()
    output_image = 'david_comparison_nk4.png'
    plt.savefig(output_image, dpi=300)
    print(f"\nSuccessfully generated and saved comparison chart as '{output_image}'")
