#!/bin/bash
# Runs inside a nukwa job (like the OnDemand kernel). Submits the same 2-node MPI
# job three ways to isolate what breaks mpirun when sbatch is called from a job.
cd ~/hpc-course-jobs
cat > diag.sbatch <<'EOS'
#!/bin/bash
#SBATCH --partition=kura-wide
#SBATCH --time=0-00:05:00
#SBATCH --cpus-per-task=1
echo "--- SLURM_ vars seen by the child job:"; env | grep -E "^SLURM_(MEM|JOB_QOS|CPUS_PER_TASK|NTASKS|TASKS_PER_NODE|NNODES|JOB_NUM_NODES|CPUS_ON_NODE)" | sort | tr "\n" " "; echo
module load gcc/12.4.0 openmpi/4.0.5
mpirun --mca orte_keep_fqdn_hostnames 1 --mca plm_base_verbose 5 /data/casch/hpc-course/env/.venv/bin/python hello.py 2>&1 | grep -iE "^rank|global|srun: error|Requested|failed|denied" | head -8
EOS
run() { echo "=== $1"; shift; jid=$("$@" --parsable --wait --nodes=2 --ntasks-per-node=2 diag.sbatch); cat slurm-$jid.out; }
run "plain nested sbatch" sbatch
run "sbatch --export=NONE" sbatch --export=NONE
run "SLURM_* stripped from submitter env" env $(env | grep -o "^SLURM_[A-Za-z_]*" | sed "s/^/-u /") sbatch
