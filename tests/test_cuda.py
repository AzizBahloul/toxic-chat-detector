import torch


def test_cuda_availability():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Current CUDA device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))

        # Test CUDA tensor operations
        x = torch.rand(5, 3)
        print("\nCPU tensor:")
        print(x)

        x = x.cuda()
        print("\nGPU tensor:")
        print(x)


if __name__ == "__main__":
    test_cuda_availability()
