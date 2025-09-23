import torch

def main():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"CUDA is available. Number of GPUs: {gpu_count}")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            print(f"GPU {i}: {name}, Compute Capability: {capability[0]}.{capability[1]}")
    else:
        print("CUDA is NOT available.")

if __name__ == "__main__":
    main()
