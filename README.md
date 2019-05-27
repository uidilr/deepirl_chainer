# Deep IRL

## Contents

Chainer implementation of Adversarial Inverse Reinforcement Learning (AIRL) and Generative Adversarial Imitation Learning (GAIL). 
The code heavily depend on the reinforcement learning package Chainerrl.

## Commands

Train and sample expert trajectory
```bash
python train_gym.py ppo --gpu -1 --env CartPole-v0 --arch FFSoftmax --steps 50000 
```

Run GAIL
```bash
python train_gym.py gail --gpu -1 --env CartPole-v0 --arch FFSoftmax --steps 100000 \
                    --load_demo ${PathOfDemonstrationNpzFile} --update-interval 128 -entropy-coef 0.01
```

Run AIRL
```bash
python train_gym.py airl --gpu -1 --env CartPole-v0 --arch FFSoftmax --steps 100000 \
                    --load_demo ${PathOfDemonstrationNpzFile} --update-interval 128 -entropy-coef 0.01
```
