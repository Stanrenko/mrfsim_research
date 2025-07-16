
ls $1*.dat | while read -r line ; do
    echo "Processing $line"

    filename=$(basename "$line")
    echo $filename
    filepath=$(dirname "$line")


    file_no_ext="${filename%.*}"

    echo ${filepath}/${file_no_ext}

    sh run_MRF_3D_lowerbody.sh ${filepath}/${file_no_ext} $2 $3

    # your code goes here
done