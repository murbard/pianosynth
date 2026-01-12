import torch
import scipy.io.wavfile
import numpy as np

def save_wav(filename: str, audio: torch.Tensor, sr: int = 44100):
    """
    Saves a 1D torch tensor to a WAV file.
    """
    # Move to CPU and detach
    audio_cpu = audio.detach().cpu()
    
    # Clip to -1.0 to 1.0 to avoid distortion
    audio_cpu = torch.clamp(audio_cpu, -1.0, 1.0)
    
    # Convert to numpy
    audio_np = audio_cpu.numpy()
    
    # Write using scipy
    # Convert to 16-bit PCM for maximum compatibility
    audio_int16 = (audio_np * 32767).astype(np.int16)
    scipy.io.wavfile.write(filename, sr, audio_int16)
