import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder_net = nn.Linear(input_dim, hidden_dim),
        self.decoder_net = nn.Linear(hidden_dim, input_dim)

    def encoder(self, x):
        return self.encoder_net(x)
    
    def decoder(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        z = self.encoder_net(x)
        decoded = self.decoder_net(z)
        return z, decoded
    
    def fit(self, states, num_epochs=10, learning_rate=1e-3, device="cpu"):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            for state in states:
                optimizer.zero_grad()
                _, compressed_state = self(state)
                loss = criterion(compressed_state, state)
                loss.backward()
                optimizer.step()