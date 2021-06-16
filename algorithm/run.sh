python main.py \
        --algorithm fedavg \
        --dataset cifar10 \
        --model cifar10 \
        --numRounds 100 \
        --evalInterval 1 \
        --clientsPerRound 10 \
        --epoch 5 \
        --batchSize 50 \
        --lr 0.1 \
        --lrDecay 0.99 \
        --decayStep 1 \
        --alpha 0.1 \
        --seed 12 \
        --cuda True \
        --mu 0.001 \
        --omega 10
