import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb

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

        wandb.log({"Loss": loss_modified.item()})

    return losses

wandb.init(project='Guided Research toy example', name="Combined Method")
device = "cuda:0"
model_normal = ToyModel().to(device)
model_modified = ToyModelModified().to(device)

# Combine parameters from both models
parameters_normal = list(model_normal.parameters())
parameters_modified = list(model_modified.parameters())

# Use a single optimizer
optimizer_normal = optim.Adam(parameters_normal, lr=0.01)
optimizer_modified = optim.Adam(parameters_modified, lr=0.01)

# Train the models
losses_combined = train(model_normal, model_modified, optimizer_normal, optimizer_modified, device, epochs=400, time_steps=4)

# Loss Landscape plot for the Original Model
plt.figure()
plt.plot(losses_combined, label='Combined')
plt.title('Loss curve - Combined Backpropagation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve_combined.png')

wandb.finish()