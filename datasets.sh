#!/bin/sh

#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH -J job
#SBATCH -o ./out/%j-0.out
#SBATCH -e ./err/%j-0.out
#SBATCH -a 0-21%5

module load python/3.7.4 gcc/8.3
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate features

echo "job started."

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ];
then
python sva.py --template base --weak lexical
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ];
then
python sva.py --template base --weak agreement
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 2 ];
then
python sva.py --template base --weak plural
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 3 ];
then
python sva.py --template hard --weak lexical
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 4 ];
then
python sva.py --template hard --weak agreement
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 5 ];
then
python sva.py --template hard --weak plural
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 6 ];
then
python sva.py --template hard --weak length
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 7 ];
then
python npi.py --weak lexical
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 8 ];
then
python npi.py --weak plural
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 9 ];
then
python npi.py --weak tense
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 10 ];
then
python npi.py --weak length
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 11 ];
then
python toy.py --true_property 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 12 ];
then
python toy.py --true_property 2
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 13 ];
then
python toy.py --true_property 3
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 14 ];
then
python toy.py --true_property 4
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 15 ];
then
python toy.py --true_property 5
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 16 ];
then
python gap.py --template base --weak length
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 17 ];
then
python gap.py --template base --weak lexical
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 18 ];
then
python gap.py --template base --weak plural
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 19 ];
then
python gap.py --template base --weak tense
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 20 ];
then
python gap.py --template hard --weak none
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 21 ];
then
python gap.py --template hard --weak length
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 22 ];
then
python gap.py --template hard --weak lexical
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 23 ];
then
python gap.py --template hard --weak plural
fi
if [ "$SLURM_ARRAY_TASK_ID" -eq 24 ];
then
python gap.py --template hard --weak tense
fi