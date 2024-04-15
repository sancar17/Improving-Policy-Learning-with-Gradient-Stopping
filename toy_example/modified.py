import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb

# Define the toy example model and simulator
class ToyModelModified(nn.Module):
    def __init__(self):
        super(ToyModelModified, self).__init__()
        self.theta1 = nn.Parameter(torch.tensor(0.5))
        self.theta2 = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x):
        x = x.detach()
        return -self.theta1 * x**2 + self.theta2 * x

def simulator(x, c):
    return x + c

# Define the training process
def train(model, optimizer, device, epochs=100, time_steps=4):
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        x0 = torch.tensor(0.3, requires_grad=True).to(device) # initial state
        y = torch.tensor(-2.0).to(device) # target state

        # Unrolling over time steps
        x = x0
        for _ in range(time_steps):
            c = model(x)
            x = simulator(x, c)

        loss = 0.5 * (x - y)**2
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

        wandb.log({"Loss": loss.item()})

    return losses

device = "cuda:0"
# Initialize the model and optimizer
model_modified = ToyModelModified().to(device)
optimizer_modified = optim.Adam(model_modified.parameters(), lr=0.01)

wandb.init(project='Guided Research toy example', name="Modified Method")

# Train the models
losses_modified = train(model_modified, optimizer_modified, device, epochs=400, time_steps=4)

# Loss Landscape plot for the Original Model
plt.figure()
plt.plot(losses_modified, label='Modified')
plt.title('Loss Curve - Modified Backpropagation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve_modified.png')

wandb.finish()