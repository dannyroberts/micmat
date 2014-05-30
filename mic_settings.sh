export MICMAT_PATH=/global/homes/r/rippel/sota/micmat

export PATH=~/anaconda/bin:$PATH
# export PYTHONPATH=$PYTHONPATH:$MICMAT_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MICMAT_PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MKLROOT/lib/intel64
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mic/coi/host-linux-release/lib
#export MIC_LD_LIBRARY_PATH=$MIC_LD_LIBRARY_PATH:$MKLROOT/lib/mic


source /opt/intel/composerxe/bin/compilervars.sh intel64 
source $MKLROOT/bin/mklvars.sh mic lp64
source /opt/intel/impi/4.1.1.036/bin64/mpivars.sh

# export OFFLOAD_REPORT=2
export MIC_ENV_PREFIX=PHI
export PHI_OMP_NUM_THREADS=240
export OFFLOAD_INIT=on_start
export PHI_USE_2MB_BUFFERS=16K
export PHI_KMP_AFFINITY="granularity=fine,balanced"