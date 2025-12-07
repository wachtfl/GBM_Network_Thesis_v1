#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH
#SBATCH --partition main ### partition name where to run a job. Use ‘main’ unless qos is required. qos partitions ‘rtx3090’ ‘rtx2080’ ‘gtx1080’
#SBATCH --time 0-10:30:00 ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name my_job ### name of the job. replace my_job with your desired job name
#SBATCH --output my_job-id-%J.out ### output log for running job - %J is the job number variable
#SBATCH --mail-user=leahwach@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=0 ### e.g.: #SBATCH --gpus=gtx_1080:1 , or rtx_2080, or rtx_3090.
##SBATCH --tasks=1 # 1 process – use for processing of few programs concurrently in a job (with srun). Use just 1 otherwise
### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
### Start your code below ####
module load anaconda ### load anaconda module
source activate my_env ### activate a conda environment, replace my_env with your conda environment
python run_main_pipeline.py --data_dir --model 'rf' --cv  'lopo' ### model: rf/dtree/logistic/svm cv: lopo/kfold (3-fold default) -n_splits