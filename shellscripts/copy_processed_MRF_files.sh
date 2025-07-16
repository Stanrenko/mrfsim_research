#set -e

#Processing whole body hybrid scan data
# $1 : Folder containing all the .dat


targetfolder="data/InVivo/3D/processing"

processedfolder=$(echo ${1} | sed 's/2_Data_Raw/3_Data_Processed/')


EXAM=-4
echo "Copying map files"
find $processedfolder -type f -name "*bart*MRF_map.pkl" | sort | while read -r line ; do

    echo $line
    echo $EXAM

    filename=$(basename "$line")
    filepath=$(dirname "$line")


    file_mrf="${filename%.*}"

    echo $file_mrf

    # if [ $EXAM -eq -4 ]
    # then
    #     EXAM=$((EXAM+1))
    #     continue
    # fi

    

    echo ${filepath}/${file_mrf}

    echo "Copying ${filepath}/${filename} in ${targetfolder}"
    cp ${filepath}/${filename} ${targetfolder}
    echo "Copied ${filepath}/${filename} in ${targetfolder}"
    EXAM=$((EXAM+1))

done



EXAM=-4
echo "Copying mha files"
find $processedfolder -type f -name "*bart*wT1.mha" | sort | while read -r line ; do

    echo $line
    echo $EXAM

    filename=$(basename "$line")
    filepath=$(dirname "$line")


    file_mrf="${filename%.*}"

    echo $file_mrf

    # if [ $EXAM -eq -4 ]
    # then
    #     EXAM=$((EXAM+1))
    #     continue
    # fi

    

    echo ${filepath}/${file_mrf}

    echo "Copying ${filepath}/${filename} in ${targetfolder}"
    cp ${filepath}/${filename} ${targetfolder}
    echo "Copied ${filepath}/${filename} in ${targetfolder}"
    EXAM=$((EXAM+1))

done


EXAM=-4
echo "Copying mha files"
find $processedfolder -type f -name "*bart*ff.mha" | sort | while read -r line ; do

    echo $line
    echo $EXAM

    filename=$(basename "$line")
    filepath=$(dirname "$line")


    file_mrf="${filename%.*}"

    echo $file_mrf

    # if [ $EXAM -eq -4 ]
    # then
    #     EXAM=$((EXAM+1))
    #     continue
    # fi

    

    echo ${filepath}/${file_mrf}

    echo "Copying ${filepath}/${filename} in ${targetfolder}"
    cp ${filepath}/${filename} ${targetfolder}
    echo "Copied ${filepath}/${filename} in ${targetfolder}"
    EXAM=$((EXAM+1))

done


EXAM=-4
echo "Copying seq param files"
find $processedfolder -type f -name "*seqParams.pkl" | sort | while read -r line ; do

    echo $EXAM

    filename=$(basename "$line")
    filepath=$(dirname "$line")


    file_mrf="${filename%.*}"

    # if [ $EXAM -eq -4 ]
    # then
    #     EXAM=$((EXAM+1))
    #     continue
    # fi

    

    echo ${filepath}/${file_mrf}

    echo "Copying ${filepath}/${filename} in ${targetfolder}"
    cp ${filepath}/${filename} ${targetfolder}
    echo "Copied ${filepath}/${filename} in ${targetfolder}"
    EXAM=$((EXAM+1))

done

