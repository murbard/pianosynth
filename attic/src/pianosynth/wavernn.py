import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveRNN(nn.Module):
    def __init__(self, hidden_size=32):
        """
        Small WaveRNN model for waveform generation.
        
        Args:
            hidden_size: Hidden size of the GRU core.
        """
        super().__init__()
        
        # Context CNN
        # Input: (B, 1, 2000)
        # Goal: Compress 2000 samples into a small context vector.
        # We use a series of strided 1D convolutions.
        
        # Conv1: 1 -> 4 channels, stride 4
        self.conv1 = nn.Conv1d(1, 4, kernel_size=31, stride=4, padding=15)
        # Output len: 2000/4 = 500
        
        # Conv2: 4 -> 8 channels, stride 4
        self.conv2 = nn.Conv1d(4, 8, kernel_size=15, stride=4, padding=7)
        # Output len: 500/4 = 125
        
        # Conv3: 8 -> 16 channels, stride 4
        self.conv3 = nn.Conv1d(8, 16, kernel_size=7, stride=4, padding=3)
        # Output len: 125/4 = 31
        
        # Global Average Pooling to get fixed size vector from remaining temporal dim
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.context_dim = 16
        
        # RNN Core
        # Input: Context(16) + Velocity(1) + LogPitch(1) + Time(1) = 19
        self.input_dim = self.context_dim + 3
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(self.input_dim, self.hidden_size, batch_first=True)
        
        # Output Head
        # Hidden(32) -> Mu(1), LogSigma(1)
        self.proj = nn.Linear(self.hidden_size, 2)
        
    def forward(self, past_samples, velocity, log_pitch, time, hidden_state):
        """
        Single step forward pass (Inference).
        """
        B = past_samples.size(0)
        
        # 1. Context CNN
        x = past_samples.unsqueeze(1) # (B, 1, 2000)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(2) # (B, 16)
        
        # 2. RNN Input
        if velocity.dim() == 1: velocity = velocity.unsqueeze(1)
        if log_pitch.dim() == 1: log_pitch = log_pitch.unsqueeze(1)
        if time.dim() == 1: time = time.unsqueeze(1)
        
        rnn_in = torch.cat([x, velocity, log_pitch, time], dim=1) # (B, 19)
        
        # 3. RNN Step using nn.GRU
        # Input (B, 1, 19), Hidden (1, B, 32)
        rnn_in_seq = rnn_in.unsqueeze(1)
        if hidden_state is None:
            h_in = None
        else:
            h_in = hidden_state.unsqueeze(0)
            
        _, h_out = self.gru(rnn_in_seq, h_in)
        
        new_hidden = h_out.squeeze(0) # (B, 32)
        
        # 4. Output
        out = self.proj(new_hidden)
        mu, log_sigma = out.chunk(2, dim=1)
        sigma = torch.exp(log_sigma)
        
        return mu, sigma, new_hidden

    def forward_sequence(self, past_samples_seq, velocity_seq, log_pitch_seq, time_seq, hidden_state=None):
        """
        Sequence forward pass (Training).
        
        Args:
            past_samples_seq: (B, T, 2000)
            velocity_seq: (B, T, 1) or (B, 1)
            log_pitch_seq: (B, T, 1) or (B, 1)
            time_seq: (B, T, 1)
            hidden_state: (1, B, hidden_size) or None
            
        Returns:
            mus: (B, T, 1)
            sigmas: (B, T, 1)
        """
        B, T, _ = past_samples_seq.shape
        
        # 1. Parallel CNN
        # Reshape to (B*T, 1, 2000)
        flat_past = past_samples_seq.view(B * T, 1, 2000)
        x = F.relu(self.conv1(flat_past))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(2) # (B*T, 16)
        
        # Reshape back to (B, T, 16)
        ctx_seq = x.view(B, T, self.context_dim)
        
        # 2. Prepare Inputs
        if velocity_seq.dim() == 2: velocity_seq = velocity_seq.unsqueeze(1).expand(B, T, 1)
        if log_pitch_seq.dim() == 2: log_pitch_seq = log_pitch_seq.unsqueeze(1).expand(B, T, 1)
        # Ensure time is (B, T, 1)
        if time_seq.dim() == 2: time_seq = time_seq.unsqueeze(2)
        
        rnn_in = torch.cat([ctx_seq, velocity_seq, log_pitch_seq, time_seq], dim=2) # (B, T, 19)
        
        # 3. RNN Sequence
        out_seq, _ = self.gru(rnn_in, hidden_state) # (B, T, hidden)
        
        # 4. Output
        out = self.proj(out_seq) # (B, T, 2)
        mus, log_sigmas = out.chunk(2, dim=2)
        sigmas = torch.exp(log_sigmas)
        
        return mus, sigmas


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
