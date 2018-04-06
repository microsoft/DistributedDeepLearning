
# download the benchmark scripts
cd $AZ_BATCHAI_JOB_TEMP
git clone https://github.com/alsrgv/benchmarks
cd benchmarks
git checkout horovod_v2

# install horovod
source /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/bin/mpivars.sh
# pip install horovod absl-py