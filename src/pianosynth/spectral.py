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

def diff_piano_render(freqs, tau_s, tau_f, amps, w_curve, dur_samples, sr=44100, reverb_wet=None, reverb_decay=None):
    """
    Fully differentiable synthesis for a batch of partials with double decay + 3 strings.
    freqs: [Batch, N, 3] (Center, Left, Right)
    tau_s: [Batch, N] (Broadcasts)
    tau_f: [Batch, N] (Broadcasts)
    amps: [Batch, N, 3] (Base amps * string_mask)
    w_curve: [Batch, N]
    dur_samples: int
    reverb_wet: [Batch] or scalar, optional mix amount
    reverb_decay: [Batch] or scalar, optional reverb time
    """
    device = freqs.device
    B, N, S = freqs.shape
    
    t = torch.arange(dur_samples, device=device).float() / sr # [T]
    
    # Time envelopes
    tau_s = tau_s.clamp(min=1e-4) # [Batch, N]
    tau_f = tau_f.clamp(min=1e-4)
    
    # [Batch, N, 1, T] 
    # Use explicit dimensions
    E_slow = torch.exp(-t.reshape(1, 1, 1, -1) / tau_s.unsqueeze(2).unsqueeze(3))
    E_fast = torch.exp(-t.reshape(1, 1, 1, -1) / tau_f.unsqueeze(2).unsqueeze(3))
    
    # Mixing
    # [Batch, N, 3, 1]
    # Amps passed in already include string mask and A0 logic.
    w = w_curve.unsqueeze(2).unsqueeze(3) # [B, N, 1, 1]
    
    A_fast = amps.unsqueeze(3) * w
    A_slow = amps.unsqueeze(3) * (1.0 - w)
    
    env = A_fast * E_fast + A_slow * E_slow # [B, N, 3, T]
    
    # Phases
    # 3 strings should have independent phase? 
    # In synth.py: 0.15 * rand.
    # [Batch, N, 3, 1]
    phi = torch.rand(B, N, S, 1, device=device) * 0.15 * 2 * torch.pi
    
    # Sinusoids
    # [B, N, 3, T]
    sinusoid = torch.sin(2 * torch.pi * freqs.unsqueeze(3) * t.reshape(1, 1, 1, -1) + phi)
    
    # Sum over strings (dim 2) AND partials (dim 1)
    y = (env * sinusoid).sum(dim=2).sum(dim=1) # [B, T]
    
    # Attack Envelope
    rise = 1.0 - torch.exp(-t / 0.005)
    y = y * rise.unsqueeze(0)

    # Reverb
    if reverb_wet is not None and reverb_decay is not None:
        # Generate Impulse Response (Exponential Decay Noise)
        # Unique IR per batch or shared? 
        # Ideally unique if parameters vary, but 'noise' can be constant seed?
        # Let's use constant seed for noise texture to avoid noise gradient variance,
        # but apply variable envelope.
        
        # We need independent IRs if we want independent randomness, but for "room" simulation 
        # usually 1 room per batch. If batch has multiple notes, they share room? 
        # But we optimize single notes mostly.
        # Let's generate [1, T] noise for efficiency if params are same, else [B, T].
        
        if torch.is_tensor(reverb_decay):
            dk = reverb_decay.view(-1, 1)
        else:
            dk = torch.tensor(reverb_decay, device=device).view(1, 1)
            
        noise = torch.randn(1, dur_samples, device=device) # Shared noise texture
        
        # Envelope: exp(-t / decay)
        # t is [T]
        env_ir = torch.exp(-t.unsqueeze(0) / (dk + 1e-6))
        
        ir = noise * env_ir # [B, T]
        
        # Normalize Energy of IR to 1 (or something consistent)
        ir_energy = torch.norm(ir, dim=1, keepdim=True) + 1e-6
        ir = ir / ir_energy
        
        # Convolution via FFT
        # Padding: linear convolution requires T + T - 1 length. 
        # But we want output same size? Or reverb tail extending?
        # Usually wet signal extends. But our data is fixed length.
        # We'll maintain length 'dur_samples'.
        
        n_fft = 2 * dur_samples 
        # Round up to power of 2 for speed?
        
        Y = torch.fft.rfft(y, n=n_fft, dim=1)
        H = torch.fft.rfft(ir, n=n_fft, dim=1)
        
        # Broadcast H if B=1 and Y has B>1? 
        # ir is [B, T] or [1, T]. PyTorch broadcasts.
        
        Y_wet = torch.fft.irfft(Y * H, n=n_fft, dim=1)
        
        # Crop
        y_wet = Y_wet[:, :dur_samples]
        
        # Mix
        if torch.is_tensor(reverb_wet):
            wet = reverb_wet.view(-1, 1)
        else:
            wet = torch.tensor(reverb_wet, device=device).view(1, 1)
            
        y = y + wet * y_wet
    
    return y
