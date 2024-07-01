
find $1 -type f -name "*.dat" | while read -r line ; do
    echo "Processing $line"

    filename=$(basename "$line")
    echo $filename
    filepath=$(dirname "$line")


    file_no_ext="${filename%.*}"

    echo ${filepath}/${file_no_ext}

    sh run_MRF_mask_ROI.sh ${filepath}/${file_no_ext}

    # your code goes here
done