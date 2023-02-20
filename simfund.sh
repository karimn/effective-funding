#!/bin/sh

SAVE_DIR=temp-data

system76-power profile performance

#julia --project=. simfund.jl $SAVE_DIR/sim_500.jls --append \
#    --numprocs=11 \
#    --numprograms=10 --numsim=5 --numsteps=15 \
#    --alpha=0.25 --pftdpw-iter=500 --k-state=10 --use-dgp-priors 

julia --project=. simfund.jl $SAVE_DIR/sim_1000.jls --append \
    --numprocs=11 \
    --numprograms=10 --numsim=5 --numsteps=15 \
    --alpha=0.25 --pftdpw-iter=1000 --k-state=10 --use-dgp-priors \
    --reward-only
    
