#!/bin/sh
DATE=`date +%Y-%m-%d`

python job.py --experiment finetune
python job.py --experiment probing

echo $DATE
# https://hpc.nih.gov/docs/job_dependencies.html
almostjid1=$(sbatch jobs/finetune-$DATE.sh)
arr=($almostjid1)
jid1=${arr[3]}
sbatch --dependency=afterok:$jid1 jobs/probing-$DATE.sh