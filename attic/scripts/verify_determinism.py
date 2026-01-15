import torch
from pianosynth.optimization_batch import PianoParamPerKey
from pianosynth.spectral import MultiResSTFTLoss, diff_piano_render
from pianosynth.physics import calculate_partials
from pathlib import Path

PROCESSED_DATA_DIR = Path("data/processed")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    m = 60
    d_idx = 1 # mf
    
    model = PianoParamPerKey(device=device)
    loss_fn = MultiResSTFTLoss(device=device)
    
    # Load Audio
    f = PROCESSED_DATA_DIR / "60_mf.pt"
    if not f.exists():
        print("Audio not found")
        return
    audio_gt = torch.load(f).float().to(device)
    CLIP_LEN = 44100 * 2
    if len(audio_gt) > CLIP_LEN: audio_gt = audio_gt[:CLIP_LEN]
    else: return
    if audio_gt.ndim == 1: audio_gt = audio_gt.unsqueeze(0)
    
    m_t = torch.tensor([m], device=device).float()
    d_t = torch.tensor([d_idx], device=device).long()
    
    print(f"Checking determinism for M={m}...", flush=True)
    
    losses = []
    
    for i in range(10):
        # Reset Seed
        seed = 42 + m + d_idx * 1000
        torch.manual_seed(seed)
        
        with torch.no_grad():
            overrides = model(m_t, d_t)
            phys = calculate_partials(m_t, overrides, device=device)
            y = diff_piano_render(
                phys["freqs"], phys["tau_s"], phys["tau_f"],
                phys["amps"], phys["w_curve"], CLIP_LEN,
                reverb_wet=phys.get("reverb_wet"), reverb_decay=phys.get("reverb_decay")
            )
            if y.ndim == 1: y = y.unsqueeze(0)
            loss = loss_fn(y, audio_gt).item()
            losses.append(loss)
            
        print(f"Run {i}: {loss:.6f}", flush=True)
        
    # Check variance
    first = losses[0]
    if all(x == first for x in losses):
        print("FULLY DETERMINISTIC", flush=True)
    else:
        print("NON-DETERMINISTIC", flush=True)
        print(losses, flush=True)

if __name__ == "__main__":
    main()
