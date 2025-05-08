import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version (PyTorch compiled with): {torch.version.cuda}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    try:
        print("Trying to use GPU 0...")
        device0 = torch.device("cuda:1")
        a = torch.randn(3, 3, device=device0)
        b = torch.randn(3, 3, device=device0)
        c = a @ b
        print("Matrix multiplication on GPU 0 successful:")
        print(c)

        if torch.cuda.device_count() > 1:
            print("\nTrying to use nn.DataParallel with 2 GPUs (if available)...")
            # A simple model to test DataParallel
            model = torch.nn.Linear(10, 1).to(device0) # Base model on GPU 0
            try:
                dp_model = torch.nn.DataParallel(model) # Default uses all available GPUs
                print("nn.DataParallel wrapper created.")
                # Test a forward pass
                input_tensor = torch.randn(4, 10).to(device0) # Batch size 4
                output = dp_model(input_tensor)
                print("Forward pass with nn.DataParallel successful.")
                print(output)
            except Exception as e_dp:
                print(f"Error during nn.DataParallel test: {e_dp}")

    except Exception as e:
        print(f"An error occurred during minimal CUDA test: {e}")
else:
    print("CUDA not available.")