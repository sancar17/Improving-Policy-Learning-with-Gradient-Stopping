import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import threading
import psutil
import GPUtil
import time

class ToyModelModified(nn.Module):
    def __init__(self):
        super(ToyModelModified, self).__init__()
        self.theta1 = nn.Parameter(torch.tensor(0.5))
        self.theta2 = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x):
        x = x.detach()
        return -self.theta1 * x**2 + self.theta2 * x
    
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.theta1 = nn.Parameter(torch.tensor(0.5))
        self.theta2 = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x):
        return -self.theta1 * x**2 + self.theta2 * x

def simulator(x, c):
    return x + c

# Define the training process
def train(model_normal, model_modified, optimizer_normal, optimizer_modified, device, epochs=200, time_steps=4):
    losses = []
    for epoch in range(epochs):

        if(model_normal.theta1 != model_modified.theta1):
            print("Epoch:", epoch+1)
            print("Theta 1 not equal")
            exit()
        if(model_normal.theta2 != model_modified.theta2):
            print("Epoch:", epoch+1)
            print("Theta 2 not equal")
            exit()
        
        optimizer_normal.zero_grad()
        optimizer_modified.zero_grad()

        x0 = torch.tensor(0.3, requires_grad=True).to(device)  # initial state
        y = torch.tensor(-2.0).to(device)  # target state

        # Unrolling over time steps
        x_normal = x0
        x_modified = x0
        for _ in range(time_steps):
            c_normal = model_normal(x_normal)
            x_normal = simulator(x_normal, c_normal)

            c_modified = model_modified(x_modified)
            x_modified = simulator(x_modified, c_modified)

        loss_normal = 0.5 * (x_normal - y)**2
        loss_modified = 0.5 * (x_modified - y)**2

        # Backward pass for both models
        loss_normal.backward(retain_graph=True)
        loss_modified.backward()

        # Combine gradients
        for param_normal, param_modified in zip(model_normal.parameters(), model_modified.parameters()):
            if param_normal.grad is not None and param_modified.grad is not None:
                combined_grad = param_modified.grad.clone()
                different_sign = torch.sign(param_normal.grad) != torch.sign(combined_grad)
                #print(param_normal.grad, combined_grad)
                combined_grad[different_sign] = 0
                param_normal.grad = combined_grad
                param_modified.grad = combined_grad

        # Update both models
        optimizer_normal.step()
        optimizer_modified.step()
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
model_normal = ToyModel().to(device)
model_modified = ToyModelModified().to(device)

# Combine parameters from both models
parameters_normal = list(model_normal.parameters())
parameters_modified = list(model_modified.parameters())

# Use a single optimizer
optimizer_normal = optim.Adam(parameters_normal, lr=0.01)
optimizer_modified = optim.Adam(parameters_modified, lr=0.01)

# Start monitoring
monitor_thread_running = True
monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()

# Train the models
losses_combined = train(model_normal, model_modified, optimizer_normal, optimizer_modified, device, epochs=4000, time_steps=4)

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
plt.savefig("combined_monitoring.png")