import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import threading
import psutil
import GPUtil
import time
    
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.theta1 = nn.Parameter(torch.tensor(0.5))
        self.theta2 = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x, modified=False):
        if(modified==False):
            return -self.theta1 * x**2 + self.theta2 * x
        
        x = x.detach()
        return -self.theta1 * x**2 + self.theta2 * x

def simulator(x, c):
    return x + c

# Define the training process
def train(model, optimizer, device, epochs=200, time_steps=4):
    losses = []

    for epoch in range(epochs):
        
        optimizer.zero_grad()

        x0 = torch.tensor(0.3, requires_grad=True).to(device)
        y = torch.tensor(-2.0).to(device)
        x_normal = x0
        x_modified = x0

        for _ in range(time_steps):
            c_normal = model(x_normal)
            x_normal = simulator(x_normal, c_normal)
        
        loss_normal = 0.5 * (x_normal - y)**2
        loss_normal.backward(retain_graph=True)

        # Store gradients
        normal_gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                normal_gradients[name] = param.grad.clone()

        # Zero out gradients
        optimizer.zero_grad()

        for _ in range(time_steps):
            c_modified = model(x_modified, modified=True)
            x_modified = simulator(x_modified, c_modified)

        loss_modified = 0.5 * (x_modified - y)**2        
        loss_modified.backward()

        combined_gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                combined_gradients[name] = param.grad.clone()
        
        optimizer.zero_grad()

        # Combine gradients
        for name, param in model.named_parameters():
            if normal_gradients[name] is not None and combined_gradients[name] is not None:
                #print(normal_gradients[name], combined_gradients[name])
                if(torch.sign(normal_gradients[name]) != torch.sign(combined_gradients[name])):
                    combined_gradients[name] = torch.zeros_like(combined_gradients[name])
            param.grad = combined_gradients[name]

        # Update both models
        optimizer.step()
        losses.append(loss_modified.item())


    return losses

# Global lists to store the usage stats
cpu_usage = []
ram_usage = []
gpu_usage = []

def monitor_resources(interval=1):
    """Monitors CPU, RAM, and GPU usage and appends stats to global lists."""
    while monitor_thread_running:
        cpu_usage.append(psutil.cpu_percent())
        ram_usage.append(psutil.virtual_memory().percent)
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage.append(round(gpus[0].load * 100, 2))
        else:
            gpu_usage.append(0)
        time.sleep(interval)

device = "cuda:0"

model = ToyModel().to(device)

parameters = list(model.parameters())

optimizer = optim.Adam(parameters, lr=0.01)

# Start monitoring
monitor_thread_running = True
monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()

# Train the models
losses_combined = train(model, optimizer, device, epochs=4000, time_steps=4)

# Stop monitoring
monitor_thread_running = False
monitor_thread.join()

# Plotting the resource usage
plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
plt.plot(cpu_usage, label='CPU Usage')
plt.title('CPU Usage')
plt.ylabel('Percentage')

plt.subplot(3, 1, 2)
plt.plot(ram_usage, label='RAM Usage')
plt.title('RAM Usage')
plt.ylabel('Percentage')

plt.subplot(3, 1, 3)
if gpu_usage:
    plt.plot(gpu_usage, label='GPU Usage')
    plt.title('GPU Usage')
    plt.ylabel('Percentage')
else:
    plt.text(0.5, 0.5, 'No GPU Usage Data', horizontalalignment='center')

plt.xlabel('Time (seconds)')
plt.tight_layout()
plt.show()
plt.savefig("combined_one_model_monitoring.png")
