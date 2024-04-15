import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Define the ToyModel and ToyModelModified classes
class ToyModel(nn.Module):
    def __init__(self, theta1, theta2):
        super(ToyModel, self).__init__()
        self.theta1 = nn.Parameter(torch.tensor(theta1))
        self.theta2 = nn.Parameter(torch.tensor(theta2))

    def forward(self, x):
        return self.theta1 * x**2 + self.theta2 * x

class ToyModelModified(nn.Module):
    def __init__(self, theta1, theta2):
        super(ToyModelModified, self).__init__()
        self.theta1 = nn.Parameter(torch.tensor(theta1))
        self.theta2 = nn.Parameter(torch.tensor(theta2))

    def forward(self, x):
        x = x.detach()  # Detach x to prevent gradients from flowing into it
        return -self.theta1 * x**2 + self.theta2 * x

# Define the simulator and loss calculation function
def simulator(x, c):
    return x + c

def calculate_loss(model, x0, y, time_steps=4):
    x = x0
    for _ in range(time_steps):
        c = model(x)
        x = simulator(x, c)
    return 0.5 * (x - y)**2

# Function to calculate the gradients
def calculate_gradients(theta1, theta2, x0, y, time_steps=4):
    model_normal = ToyModel(theta1, theta2)
    model_modified = ToyModelModified(theta1, theta2)

    # Calculate the loss
    loss_normal = calculate_loss(model_normal, x0, y, time_steps)
    loss_modified = calculate_loss(model_modified, x0, y, time_steps)

    # Zero the gradients before backward pass
    model_normal.zero_grad()
    model_modified.zero_grad()

    # Perform the backward pass to get the gradients
    loss_normal.backward()
    loss_modified.backward()

    # Extract the gradients
    grad_normal = (-model_normal.theta1.grad.item(), -model_normal.theta2.grad.item())
    grad_modified = (-model_modified.theta1.grad.item(), -model_modified.theta2.grad.item())

    return grad_normal, grad_modified

# Function to calculate the loss landscape for a given model
def calculate_loss_landscape(model_class, theta1_values, theta2_values, x0, y):
    loss_landscape = np.zeros((theta1_values.size, theta2_values.size))
    for i, theta1 in enumerate(theta1_values):
        for j, theta2 in enumerate(theta2_values):
            model = model_class(theta1, theta2)
            loss_landscape[i, j] = calculate_loss(model, x0, y).item()
    return loss_landscape.T  # Transpose to match the orientation of the Theta grid

# Create a grid of parameter values
theta1_values = np.linspace(-5, 5, 200)
theta2_values = np.linspace(-5, 5, 200)
Theta1, Theta2 = np.meshgrid(theta1_values, theta2_values)

# Initial state and target state
x0 = torch.tensor(-0.3, requires_grad=True, dtype=torch.float32)
y = torch.tensor(2.0, dtype=torch.float32)

model = ToyModel(0.5, 0.2)
loss = calculate_loss(model, x0, y).item()

# Calculate the loss landscapes for the regular and modified models
Loss_Regular = calculate_loss_landscape(ToyModel, theta1_values, theta2_values, x0, y)
Loss_Modified = calculate_loss_landscape(ToyModelModified, theta1_values, theta2_values, x0, y)

# Initialize arrays to store gradients
Regular_Grad = np.zeros(Theta1.shape + (2,))
Modified_Grad = np.zeros(Theta1.shape + (2,))

# Calculate gradients for each point in the grid
for i in range(Theta1.shape[0]):
    for j in range(Theta1.shape[1]):
        theta1 = Theta1[i, j]
        theta2 = Theta2[i, j]
        reg_grad, mod_grad = calculate_gradients(theta1, theta2, x0, y)
        Regular_Grad[i, j] = reg_grad
        Modified_Grad[i, j] = mod_grad

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Normalize color scale based on the range of the loss values
norm = colors.LogNorm(vmin=1e-4, vmax=1e4)
cmap = 'Blues_r'

# Plotting the loss landscape for the regular model
contour_regular = axs[0].pcolor(Theta1, Theta2, Loss_Regular, cmap=cmap, norm=norm)
axs[0].set_title('a) Loss Landscape')
axs[0].set_xlabel(r'$\theta_1$')
axs[0].set_ylabel(r'$\theta_2$')

# Plotting the loss landscape and overlaying streamlines for the regular gradient
axs[1].contourf(Theta1, Theta2, Loss_Regular, levels=100, cmap=cmap, norm=norm, alpha=0.7)
axs[1].streamplot(Theta1, Theta2, Regular_Grad[..., 0], Regular_Grad[..., 1], color='r', density=1)
axs[1].set_title('b) Gradient Field')
axs[1].set_xlabel(r'$\theta_1$')

# Plotting the loss landscape and overlaying streamlines for the modified gradient
axs[2].contourf(Theta1, Theta2, Loss_Modified, levels=100, cmap=cmap, norm=norm, alpha=0.7)
axs[2].streamplot(Theta1, Theta2, Modified_Grad[..., 0], Modified_Grad[..., 1], color='y', density=1)
axs[2].set_title('c) Modified Field')
axs[2].set_xlabel(r'$\theta_1$')

# Adding a colorbar to the last plot to indicate the loss scale
fig.colorbar(contour_regular, ax=axs[2], format='%.0e')

plt.tight_layout()
plt.savefig('./Figure.png')
plt.show()
