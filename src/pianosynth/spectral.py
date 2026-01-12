import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[2048, 1024, 512, 128], device="cpu"):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.device = device
        
    def forward(self, y_pred, y_true):
        """
        y_pred, y_true: [Batch, Time]
        """
        loss = 0.0
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            win = torch.hann_window(n_fft, device=self.device)
            
            # STFT
            # [Batch, F, T, 2] -> [Batch, F, T] (mag)
            S_pred = torch.stft(y_pred, n_fft, hop_length=hop, window=win, return_complex=True).abs()
            S_true = torch.stft(y_true, n_fft, hop_length=hop, window=win, return_complex=True).abs()
            
            # Spectral Convergence Loss
            sc_loss = torch.norm(S_true - S_pred, p="fro") / (torch.norm(S_true, p="fro") + 1e-6)
            
            # Log Magnitude Loss
            log_S_pred = torch.log(S_pred + 1e-6)
            log_S_true = torch.log(S_true + 1e-6)
            
            # Weighted by Linear Amplitude (Energy)
            # This prevents fitting the noise floor loop (spectral soup)
            # We focus mainly on the peaks.
            weights = S_true
            weights = weights / (weights.mean() + 1e-8) # Normalize scale
            
            mag_loss = torch.mean(torch.abs(log_S_pred - log_S_true) * weights)
            
            loss += sc_loss + mag_loss
            
        return loss / len(self.fft_sizes)

def diff_piano_render(freqs, tau_s, tau_f, amps, w_curve, dur_samples, sr=44100):
    """
    Fully differentiable synthesis for a batch of partials with double decay.
    freqs: [Batch, N]
    tau_s: [Batch, N] (Slow decay)
    tau_f: [Batch, N] (Fast decay)
    amps: [Batch, N] (Initial amplitudes A0)
    w_curve: [Batch, N] (Weight of fast decay)
    dur_samples: int
    """
    device = freqs.device
    B, N = freqs.shape
    
    t = torch.arange(dur_samples, device=device).float() / sr # [T]
    
    # Time envelopes
    tau_s = tau_s.clamp(min=1e-4)
    tau_f = tau_f.clamp(min=1e-4)
    
    E_slow = torch.exp(-t.reshape(1, 1, -1) / tau_s.unsqueeze(2))
    E_fast = torch.exp(-t.reshape(1, 1, -1) / tau_f.unsqueeze(2))
    
    # Mixing
    A_fast = amps.unsqueeze(2) * w_curve.unsqueeze(2)
    A_slow = amps.unsqueeze(2) * (1.0 - w_curve.unsqueeze(2))
    
    env = A_fast * E_fast + A_slow * E_slow
    
    # Phases
    # Match synth.py logic: random * 0.15 * 2pi
    # This prevents "phaser" soup (full random) but keeps some organic variance.
    phi = torch.rand(B, N, 1, device=device) * 0.15 * 2 * torch.pi
    
    sinusoid = torch.sin(2 * torch.pi * freqs.unsqueeze(2) * t.reshape(1, 1, -1) + phi)
    
    # Sum over partials
    y = (env * sinusoid).sum(dim=1)
    
    # Attack Envelope
    rise = 1.0 - torch.exp(-t / 0.005)
    y = y * rise.unsqueeze(0)
    
    return y
