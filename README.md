# RL-based Hyperparameter Scheduling for Distantly Supervised NER (Stage I)

## Overview

This project proposes a Reinforcement Learning (RL)–based module for **automated hyperparameter scheduling** in the first stage of weakly supervised Named Entity Recognition (NER), based on the BERT-assisted distant supervision framework from BOND. 

Instead of relying on traditional grid search or random search to select static training parameters, our method introduces an RL agent that **dynamically adjusts hyperparameters** such as learning rate, weight decay, and optimizer configurations during training. The agent observes real-time training metrics (e.g., F1 score, loss, confidence) and learns a policy that improves model robustness and reduces manual tuning.

**Note:** This work focuses solely on **Stage I** of the BOND framework and does not modify or extend its self-training pipeline (Stage II).

## Key Contributions

- Integration of a reinforcement learning controller for hyperparameter adaptation during distant supervision.
- Discretization of high-dimensional search space for effective application of RL algorithms.
- Reward functions based on improvements in validation F1 score, reduction in loss, and pseudo-label confidence.
- Compatible with BERT-based NER models trained on weak supervision.

## RL Algorithms Implemented

- ε-greedy
- Gaussian Thompson Sampling
- Deep Q-Network (DQN)
- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)

## Experimental Setup

- **Datasets**: conll03-distant, ontonotes5-distant, wikigold-distant, twitter-distant, webpage-distant
- **Model**: RoBERTa (and variants) from HuggingFace Transformers
- **Evaluation**: F1 score, loss, precision, and training stability
