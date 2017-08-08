#!/bin/bash
#PBS -q normalbw
#PBS -l walltime=5:00:00
#PBS -l mem=10GB
#PBS -l jobfs=0GB
#PBS -l ncpus=6
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.0.1-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
#PBS -lother=mdss

##PBS -M yongzhi.xu@student.unsw.edu.au
##PBS -m ae

module load python/2.7.11
#module load tensorflow/1.0.1-python2.7
module use /g/data3/hh5/public/modules
module load conda/analysis27
module list


python  ../outdor_data_prep_util_1.py  > out_ETH_rawh5_partB.log
