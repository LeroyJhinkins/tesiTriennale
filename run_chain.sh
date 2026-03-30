#!/bin/bash
#SBATCH -J chain_tests # run's name
#SBATCH -N 1 # request 1 node
#SBATCH -c 1 # request 1 cpu per task (here you can specifiy e.g. 10, but then you need to set pool=10 in Nautilus)
#SBATCH --mem=10GB # request 10GB
#SBATCH -t 12:00:00 # request 12 hours walltime
#SBATCH -o ../out_err_files/Out_test.txt # output file name and directory
#SBATCH -e ../out_err_files/Err_test.txt # error file name and directory
#SBATCH --mail-type=BEGIN,END,FAIL # send me an e-mail at begining/end/fail of the job, won't work for galileo
#SBATCH --mail-user=andreainveninato@galileo.mi.infn.it #Related to option above, but won't work for galileo

source ./venv/bin/activate # Activate your environment

python triangle_plot_corner CLPT 20 wedges measured 1,2 # Call it as from the command line
# run CLEFT_plot_corner CLPT 40 multipoles measured 1,2 # Call it as from the command line
# run CLEFT_fit_alt_stat CLPT 0 wedges correct 2 # Call it as from the command line

