#!/bin/bash
set -eux

# generate dataset for BC
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    python run_expert.py experts/$e.pkl $e --num_rollouts=50
done

# run BC
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    python main.py --lr=1e-4 --env=$e --epoch=50 --log-interval=10000 --rollouts=50
done

# run DAgger for different number of episodes sampled in aggregation stage
for d in 1 5 10
do
    python main.py --rollouts=50 --epoch=50 --lr=1e-4 \
    --batch-size=64 --seed=1 --env=Ant-v2 --dagger=$d \
    --iteration=10 --log-interval=1000000
done

# run DAgger for less epoch in training stage
python main.py --rollouts=50 --epoch=10 --lr=1e-4 \
    --batch-size=64 --seed=1 --env=Ant-v2 --dagger=1 \
    --iteration=10 --log-interval=1000000