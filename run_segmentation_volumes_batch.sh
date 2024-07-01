
find $1 -type f -name "*MRF_map.pkl" | while read -r line ; do
    echo "Processing $line"

    python script_recoInVivo_3D_machines.py generate_dixon_volumes_for_segmentation --filemap $line --reorient False

    # your code goes here
done