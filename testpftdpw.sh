#!/bin/sh

SAVE_DIR=../temp-data

julia --project=.. testpftdpw.jl $SAVE_DIR/sim.jls \
    --numprocs=11 \
    --numprograms=10 --numsim=10 --numsteps=10 \
    --alpha=0.25 --pftdpw-iter=500 --k-state=10 --use-dgp-priors  

#julia --project=.. testpftdpw.jl $SAVE_DIR/sim_test.jls \
#    --numprocs=11 \
#    --numprograms=10 --numsim=20 --numsteps=5 \
#    --alpha=0.25 --pftdpw-iter=500 --k-state=10 --use-dgp-priors  
    