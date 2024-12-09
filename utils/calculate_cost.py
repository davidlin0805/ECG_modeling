import time
import psutil
import torch
import os


# Fungsi untuk menghitung ukuran model
def get_model_size(model_path):
    return os.path.getsize(model_path)

# Fungsi untuk menghitung kecepatan inferensi
def test_model_inference_speed(loader, model, device, num_samples=100):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i >= num_samples:
                break
            data = data.to(device)
            model(data)
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_sample = total_time / num_samples
    return avg_time_per_sample