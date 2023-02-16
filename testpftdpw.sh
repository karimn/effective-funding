#!/bin/sh

SAVE_DIR=temp-data

#julia --project=. testpftdpw.jl $SAVE_DIR/sim_500.jls --append \
#    --numprocs=11 \
#    --numprograms=10 --numsim=5 --numsteps=15 \
#    --alpha=0.25 --pftdpw-iter=500 --k-state=10 --use-dgp-priors 

julia --project=. testpftdpw.jl $SAVE_DIR/sim_test2.jls --append \
    --numprocs=11 \
    --numprograms=10 --numsim=5 --numsteps=15 \
    --alpha=0.25 --pftdpw-iter=1000 --k-state=10 --use-dgp-priors  
    