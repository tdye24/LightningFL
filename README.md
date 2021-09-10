# LightningFL
A research-oriented federated learning framework implemented with pytorch and WandB.



Follow the following steps, easy to reproduce the experiment results.

> **Step 0:**

Register a WandB account. Refer to [WandB QuickStart](https://docs.wandb.ai/quickstart).

> **Step 1:**

```shell
cd LightningFL/
```

```shell
pip install -r requirements.txt
```

> **Step 2:**

```shell
cd LightningFL/algorithm/
```

```shell
python main.py --algorithm=fedmc --alpha=0.1 --batchSize=50 --clientsPerRound=10 --cuda=True --dataset=cifar10 --decayStep=1 --diffCo=1 --drop=big --epoch=5 --evalInterval=1 --lr=0.1 --lrDecay=0.99 --mode=concat --model=cifar10 --mu=0.0001 --numRounds=100 --omega=100 --seed=12
```

> **Step 3:**

The experiments (accuracy and loss curves) can be seen in https://wandb.ai/.

