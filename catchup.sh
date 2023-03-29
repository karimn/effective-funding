#!/bin/sh

SAVE_DIR=temp-data

system76-power profile performance

julia --project simfund.jl $SAVE_DIR/freq_evalsecond_sim_1000.jls --append \
    --catchup --states-file=$SAVE_DIR/sim_1000.jls --sim-range=101-270 \
    --plans=freq_evalsecond \
    --numprocs=11 \
    --numprograms=10 --numsim=5 --numsteps=15 \
    --alpha=0.25 --pftdpw-iter=1000 --k-state=10 --use-dgp-priors \
    --reward-only
