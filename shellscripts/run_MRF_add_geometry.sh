
find $1 -type f -name "*.dat" | while read -r line ; do
    echo "Processing $line"

    filename=$(basename "$line")
    echo $filename
    filepath=$(dirname "$line")


    file_no_ext="${filename%.*}"

    echo ${filepath}/${file_no_ext}

    find $1 -type f -name "${file_no_ext}*.mha" | while read -r line2 ; do
        echo "Processing $line2"
        python script_recoInVivo_3D_machines.py getGeometry --filemha ${line2} --filename ${line} --suffix "_adjusted"

    done


    # your code goes here
done