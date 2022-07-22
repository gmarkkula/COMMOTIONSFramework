
#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=23:00:00

#Request cores
#$ -pe smp 40

#Get email at start and end of the job
#$ -m be

#Now run the job
module load anaconda
source activate base
python do_8_combined_fitting.py oVAoBEooBEvoAI


