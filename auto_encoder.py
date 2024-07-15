import torch

import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train(self, x_train, x_test, epochs=50, batch_size=32):
        """
        Train the autoencoder and return the encoder and decoder models.

        Parameters:
        - x_train: Training data
        - x_test: Testing data
        - epochs: Number of training epochs
        - batch_size: Batch size

        Returns:
        - encoder: Encoder model
        - decoder: Decoder model
        """
        # Convert data to tensors
        x_train = x_train.clone().detach().to(dtype=torch.float32)
        x_test = x_test.clone().detach().to(dtype=torch.float32)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Initialize a list to store loss values
        loss_values = []

        # Train the autoencoder
        for epoch in range(epochs):
            # Shuffle the training data
            indices = torch.randperm(x_train.size(0))
            x_train = x_train[indices]

            # Mini-batch training
            for i in range(0, x_train.size(0), batch_size):
                inputs = x_train[i:i+batch_size]

                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, inputs)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Append loss value
            loss_values.append(loss.item())

            # Print loss or other information
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        # Plotting the loss values
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='Training Loss')
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss.png')  # Save the plot as a .png file
        plt.show()  # Optionally display the plot

        # Return encoder and decoder models
        return self.encoder, self.decoder

    def encode_data(self, encoder, data):
        """
        Use the encoder to perform dimensionality reduction on the data.

        Parameters:
        - encoder: Encoder model
        - data: Data to be reduced

        Returns:
        - encoded_data: Reduced data
        """
        encoded_data = encoder(torch.tensor(data, dtype=torch.float32))
        return encoded_data.detach().numpy()