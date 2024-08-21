import torch.nn as nn

class MLP(nn.Module):
    """ This is an MLP. """
    def __init__(self):
        super(MLP, self).__init__()

        self.activation = nn.ReLU()
        self.sm = nn.Softmax(dim=1)
        
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU()
        )

        self.value_l = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.ReLU()
        )
        
        self.policy_l = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 6, bias=False),
            nn.Softmax(dim=1)
        )


    def forward(self, input):
        """ Forwarding routine. """
        x = self.shared(input)

        state_values = self.value_l(x)
        action_probs = self.policy_l(x)
        
        return action_probs, state_values