import torch.nn as nn


def Conv(i,o):
    return nn.Conv2d(in_channels=i, out_channels=o, kernel_size=3, padding=1)

class CNN(nn.Module):
    """ This is a CNN with split weights for the policy and value function. """
    def __init__(self) -> None:
        super(CNN, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = Conv(5, 7)
        self.conv2 = Conv(7, 7)
        self.conv3 = Conv(7, 7)

        self.value_l = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 7, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.ReLU()
        )

        self.policy_l = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 7, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 6, bias=False)
        )
        
        self.softmax = nn.Softmax(dim=1)

        # Output: 6 action probabilities and 1 state value

    def forward(self, input):
        """ Forwarding routine. """
        x = self.relu(self.conv1(input))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        action_probs = self.policy_l(x)
        state_value = self.value_l(x)

        action_probs = self.softmax(action_probs)

        return action_probs, state_value

