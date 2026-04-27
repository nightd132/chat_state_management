import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.encoder_net = nn.Linear(input_dim, hidden_dim)
        self.decoder_net = nn.Linear(hidden_dim, input_dim)

    def encoder(self, x):
        return self.encoder_net(x)

    def decoder(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z

    def fit(self, states, num_epochs=10, batch_size=256, learning_rate=1e-3, device="cpu"):
        self.to(device)
        self.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            total_loss = 0

            for s in states:

                s = s.detach().float()
                s = s.to(device)

                s = s.view(-1, s.size(-1))

                for i in range(0, s.size(0), batch_size):
                    batch = s[i:i+batch_size]

                    optimizer.zero_grad()

                    z = self.encoder(batch)
                    reconstructed = self.decoder(z)

                    loss = criterion(reconstructed, batch)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

