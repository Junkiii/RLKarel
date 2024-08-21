# RLKarel

![](https://github.com/Junkiii/RLKarel/blob/main/imgs/readme_img.png?raw=true)

RLKarel is a neural network training framework designed for solving the Karel Task using reinforcement learning (RL). This framework encompasses various modules for defining environments, managing data, and training models. It includes tools for evaluating and visualizing training results.

## Feature Representation
The state representation in the Karel Task environment is detailed in environment.py. The get_feature_representation function generates a feature tensor of shape b × 5 × 4 × 4, where b is the batch size. This representation supports the use of convolutional neural networks (CNNs) by providing 2D data directly. For MLPs, a Flatten layer can be used to reshape the data. The CNN has shown superior performance compared to MLPs for this task.

## Neural Network Architecture

The final model architecture includes:

1. Shared Convolutional Layers: Three 2D convolutional layers with a 3 × 3 kernel, padding of 1, and stride of 1. The first layer increases the number of channels from 5 to 7.
2. Separate MLPs: Two identical MLPs (each with four fully-connected layers with decreasing neurons) predict the state value and action probabilities. The output uses a softmax activation function for action probabilities.

## Hyperparameter
Key hyperparameters include:

1. batch_size: 256
2. γ (Discount Factor): 0.97
3. max_steps: 5 / (1 − γ)
4. learning_rate: 0.0001 (Adam optimizer is used)
5. optimizer: Adam optimizer

## Curriculum Design
Curriculum learning was not heavily relied upon in the final runs due to the effectiveness of imitation learning, large training batches, and complete episode rollouts. The framework supports training on different datasets and saving/loading models for progressive training.

## Reward Design
The reward structure includes:

1. Crash Reward: 0 (to avoid the agent from becoming overly cautious).
2. Finish Reward: 10 (encourages completion of the task).

This reward design ensures the agent focuses on completing tasks rather than just avoiding crashes.

## Imitation Learning
Imitation learning is crucial for training with larger datasets. The framework supports toggling imitation learning on and off. This feature helps balance between leveraging optimal actions and exploring new strategies.

## Data
You can contact me for the training and testing data.
