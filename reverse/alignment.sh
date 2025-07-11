#!/bin/bash

# Input parameters
ghidra_headless_path=$1
program_folder=$2
output_dir=${3:-'./output'}
timeout=${4:-1200}  # Default timeout of 10 minutes

python_script_path="$(dirname "$0")/ghidra_alignment.py"
result_folder="${output_dir}/results"

# Export variables to make them available in subshells
export ghidra_headless_path
export output_dir
export timeout
export python_script_path

# Function to extract optimization and program name from filename
extract_info() {
    local filename=$(basename "$1")
    # Extract optimization level and program name
    # Format: program-version_compiler_arch_bits_optimization_program
    local opt_prog=$(echo "$filename" | sed 's/.*_\([^_]*_[^_]*\)$/\1/')
    echo "$opt_prog"
}

# Function to process a single file
process_file() {
    local file=$1
    local file_name=$(basename "$file")
    local project_name="${file_name}_project"
    local project_folder="${output_dir}/ghidra_projects/${project_name}"
    
    # Get optimization and program info
    local opt_prog=$(extract_info "$file")
    local grouped_result_folder="${result_folder}/${opt_prog}"
    
    # Create grouped result folder
    mkdir -p "$grouped_result_folder"

    # Create a temporary project folder
    mkdir -p "$project_folder"

    # Start time measurement
    start_time=$(date +%s.%N)

    # Run Ghidra headless analyzer with timeout
    timeout --kill-after=10 "${timeout}" "$ghidra_headless_path" "$project_folder" "$project_name" -import "$file" -scriptPath "$(dirname "$python_script_path")" -postScript "$(basename "$python_script_path")" "$output_dir" "$grouped_result_folder"

    # Check if the process timed out
    if [ $? -eq 124 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S,%3N') - ERROR - Processing of $file_name timed out after $timeout seconds" >> "${output_dir}/extraction.log"
        echo "$file_name" >> "${output_dir}/timed_out_files.txt"
    else
        # End time measurement
        end_time=$(date +%s.%N)

        # Calculate execution time
        execution_time=$(echo "$end_time - $start_time" | bc)

        # Log the execution time
        echo "$(date '+%Y-%m-%d %H:%M:%S,%3N') - INFO - Successfully extracted function call information for $file_name, time: ${execution_time} seconds" >> "${output_dir}/extraction.log"
    fi

    # Remove the temporary project folder
    rm -rf "$project_folder"
}

export -f process_file
export -f extract_info

# Create necessary directories
mkdir -p "${output_dir}/ghidra_projects"
mkdir -p "$result_folder"

# Process all files in parallel
find "$program_folder" -type f | parallel --jobs $(nproc) process_file

# Clean up
rm -rf "${output_dir}/ghidra_projects"