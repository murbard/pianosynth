import torch
from pianosynth.wavernn import WaveRNN

def test_wavernn():
    model = WaveRNN(hidden_size=32)
    print(f"Total Parameters: {model.count_parameters()}")
    
    assert model.count_parameters() < 10000, f"Parameter count {model.count_parameters()} > 10000"
    
    B = 4
    past_samples = torch.randn(B, 2000)
    velocity = torch.rand(B, 1)
    log_pitch = torch.rand(B, 1)
    time = torch.rand(B, 1)
    hidden = torch.zeros(B, 32)
    
    mu, sigma, new_hidden = model(past_samples, velocity, log_pitch, time, hidden)
    
    print(f"Output Shapes: Mu {mu.shape}, Sigma {sigma.shape}, Hidden {new_hidden.shape}")
    
    print("Single Step Test Passed!")
    
    # Sequence Test
    T = 10
    past_samples_seq = torch.randn(B, T, 2000)
    velocity_seq = torch.rand(B, T, 1) # or (B, 1)
    log_pitch_seq = torch.rand(B, 1)
    time_seq = torch.rand(B, T, 1)
    
    mus, sigmas = model.forward_sequence(past_samples_seq, velocity_seq, log_pitch_seq, time_seq)
    
    print(f"Seq Output Shapes: Mus {mus.shape}, Sigmas {sigmas.shape}")
    
    assert mus.shape == (B, T, 1)
    assert sigmas.shape == (B, T, 1)
    
    print("Sequence Test Passed!")

if __name__ == "__main__":
    test_wavernn()
