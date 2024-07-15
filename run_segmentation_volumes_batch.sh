
# find $1 -type f -name "*MRF_map.pkl" | while read -r line ; do
#     echo "Processing $line"

#     python script_recoInVivo_3D_machines.py generate_dixon_volumes_for_segmentation --filemap $line --reorient False

#     # your code goes here
# done


# find $1 -type d -maxdepth 1 -name "hct*" | while read -r line ; do
#      echo "Processing $line"
# done

find $1 -maxdepth 1 -type d -name "dmu*" | while read -r folder ; do
# echo "Processing $folder"
    find ${folder} -type f -name "*raFin*.dat" | while read -r line ; do
        # echo "Processing $line"

        filename=$(basename "$line")
        # echo $filename
        filepath=$(dirname "$line")
        
        file_no_ext="${filename%.*}"

        # echo ${filepath}/${file_no_ext}

        # echo "${filepath}/processed/${file_no_ext}_oop.mha"

        if [ -f "${filepath}/processed/${file_no_ext}_oop.mha" ]; then
            echo "${filepath}/processed/${file_no_ext}_oop.mha exists"
            continue
        fi

        find ${filepath} -type f -name "${file_no_ext}*MRF_map.pkl" | while read -r line2 ; do
            # continue
            echo "################### Processing $line2 #######################################"
            python script_recoInVivo_3D_machines.py generate_dixon_volumes_for_segmentation --filemap ${line2} --filename ${line}

        done


        # your code goes here
    done
done