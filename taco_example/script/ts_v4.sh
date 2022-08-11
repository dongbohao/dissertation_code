#!/bin/bash
#$ -l h_rt=6:00:00  #time needed
#$ -pe smp 2 #number of cores
#$ -l rmem=8G #number of memery
#$ -l gpu=1 #gpu
#$ -o /home/acq21bd/Output/g_v4.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M bdong8@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary
module load apps/python/conda
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176


source activate /data/acq21bd/pytorch

export LD_LIBRARY_PATH=/data/acq21bd/pytorch/lib:$LD_LIBRARY_PATH

cd /home/acq21bd/dissertation_code/taco_example

python glottal_mel_model_v4.py
