import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, head_dim, d_state, hidden_dim):
        """Create an autoencoder for flattened per-head recurrent states."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim   = head_dim
        self.d_state    = d_state
        self.input_dim  = head_dim * d_state  # 64*128 = 8192

        self.encoder_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim)

        )
        self.decoder_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.input_dim)
        )

    def encoder(self, x):
        """Map flattened state vectors into the latent representation."""
        return self.encoder_net(x)

    def decoder(self, z):
        """Reconstruct flattened state vectors from latent representations."""
        out = self.decoder_net(z)
        return out

    def forward(self, x):
        """Encode and reconstruct a batch of state vectors."""
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z

    def fit(self, states, num_epochs=10, batch_size=256, learning_rate=1e-3, device="cpu"):
        """Train the autoencoder and return the average loss per epoch."""
        self.to(device)
        self.train()

        # Flatten to [num_samples * heads, head_dim * d_state]
        num_samples = states.shape[0]
        data = states.detach().float()
        data = data.view(-1, self.input_dim).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        loss_history = []

        for epoch in range(num_epochs):
            perm = torch.randperm(data.size(0))
            data = data[perm]

            epoch_loss  = 0.0
            num_batches = 0

            for i in range(0, data.size(0), batch_size):
                batch = data[i:i + batch_size]

                optimizer.zero_grad()
                z            = self.encoder(batch)
                reconstructed = self.decoder(z)
                loss         = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                epoch_loss  += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            print(f"  Epoch [{epoch+1:>3}/{num_epochs}] Loss: {avg_loss:.6f}")

        return loss_history