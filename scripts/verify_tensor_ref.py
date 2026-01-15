import torch

def main():
    t = torch.randn(10)
    print(f"Original: {t[0]}")
    
    # Simulate get_params
    val = t[0]
    
    # Simulate Injection
    t[0] = 999.0
    
    print(f"After Mod: {t[0]}")
    print(f"Saved Val: {val}")
    
    if val == 999.0:
        print("CRITICAL: Saved value mutated! It is a view!")
    else:
        print("SAFE: Saved value is independent.")

if __name__ == "__main__":
    main()
