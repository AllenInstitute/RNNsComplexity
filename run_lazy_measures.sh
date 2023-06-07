#!/bin/bash

for tr in 1
do
    for nNN_idx in 0 1 2 3 4 5
    do  
        jobid="run_nNNidx"$nNN_idx"_tr"$tr 
        echo '#!/bin/bash'>subjob.bash
        echo '#SBATCH -p braintv'>>subjob.bash
        echo '#SBATCH -J '${jobid//./-}>>subjob.bash
        echo '#SBATCH --mail-type=END,FAIL'>>subjob.bash
        echo '#SBATCH --mail-user=helena.liu@alleninstitute.org'>>subjob.bash
        echo '#SBATCH --nodes=1 --cpus-per-task=20'>>subjob.bash
        echo '#SBATCH --mem=10gb --time=1:00:00'>>subjob.bash
        echo '#SBATCH --output=/allen/programs/celltypes/workgroups/mousecelltypes/YHL/logs/'${jobid//./-}'.log'>>subjob.bash
        echo 'cd /allen/programs/celltypes/workgroups/mousecelltypes/YHL/src_code/RNNsComplexity/'>>subjob.bash
        echo 'python3 example_laziness.py --nNN_idx '$nNN_idx' 1>zzzz1_'${jobid}' 2>zzzz2_'${jobid}>>subjob.bash
        echo '...'
        sleep 1
        wait
        sbatch subjob.bash
        echo 'Job: '$jobid' '
    done 
done


#Clean up
rm subjob.bash