#!/bin/bash

# Get path to input file (passed as first argument or default to folders.txt)
input_file="${1:-folders.txt}"

# Resolve full path to input file
input_file=$(realpath "$input_file")
echo $input_file
# Extract directory containing the input file
base_dir=$(dirname "$input_file")

# Check that the input file exists
if [[ ! -f "$input_file" ]]; then
    echo "Error: File '$input_file' not found."
    exit 1
fi

# Read the file line by line
while IFS= read -r rel_folder || [[ -n "$rel_folder" ]]; do

    # Skip empty lines and lines starting with '#'
    [[ -z "$rel_folder" || "$rel_folder" == \#* ]] && continue

    # Compute the absolute path of the folder
    folder_abs=$(realpath "$base_dir/$rel_folder")

    # Call the function or script
    sh shellscripts/run_MRF_WholeBody_axial_new.sh "$folder_abs"

done < "$input_file"
