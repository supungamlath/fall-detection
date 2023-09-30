import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self, input_dim=5, num_layers=2, hidden_dim=256, dropout=0.2, num_classes=1
    ):
        super(LSTMModel, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # LSTM layer
        self.LSTM = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=dropout
            if self.num_layers > 1
            else 0,  # Apply dropout only for multiple layers
            batch_first=True,
        )

        # Fully connected layer
        self.fc1 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x, hidden_state=None):
        # Flatten LSTM parameters
        self.LSTM.flatten_parameters()

        # LSTM forward pass
        lstm_out, hidden_state = self.LSTM(x, hidden_state)

        # Get the output from the last time step
        last_time_step_output = lstm_out[:, -1, :]

        # Fully connected layer
        output = self.fc1(last_time_step_output)

        return output, hidden_state
