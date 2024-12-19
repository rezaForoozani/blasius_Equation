import numpy as np
import matplotlib
# Set the backend to TkAgg to avoid GTK issues
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Define parameters
l1 = 0
l2 = 1
h = 0.1
eta = np.arange(0, 10, h)
num_points = len(eta)
F = np.zeros((num_points, 3))

# Shooting method
while abs(F[-1, 1] - 1) > 0.00001:
    shot = 0.5 * (l1 + l2)
    F[0, :] = [0, 0, shot]
    
    # 4th order Runge-Kutta method
    for i in range(num_points - 1):
        f = F[i, :]
        k1 = h * np.array([f[1], f[2], -0.5 * f[0] * f[2]])
        f = F[i, :] + k1 / 2
        k2 = h * np.array([f[1], f[2], -0.5 * f[0] * f[2]])
        f = F[i, :] + k2 / 2
        k3 = h * np.array([f[1], f[2], -0.5 * f[0] * f[2]])
        f = F[i, :] + k3
        k4 = h * np.array([f[1], f[2], -0.5 * f[0] * f[2]])
        F[i + 1, :] = F[i, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    if F[-1, 1] < 1:
        l1 = shot
    else:
        l2 = shot

# Plotting the results
plt.figure(figsize=(12, 8))

plt.subplot(1, 4, 1)
plt.plot(F[:, 0], eta)
plt.xlabel('f')
plt.ylabel(r'$\eta$')

plt.subplot(1, 4, 2)
plt.plot(F[:, 1], eta)
plt.xlabel("f'")
plt.ylabel(r'$\eta$')

plt.subplot(1, 4, 3)
plt.plot(F[:, 2], eta)
plt.xlabel("f''")
plt.ylabel(r'$\eta$')

plt.subplot(1, 4, 4)
plt.plot(-0.5 * F[:, 0] * F[:, 2], eta)
plt.xlabel("f'''")
plt.ylabel(r'$\eta$')

plt.tight_layout()

# Save the figure
plt.savefig('blasius_solution.png', dpi=300)  # Save as PNG with 300 dpi
plt.close()  # Close the figure to free up memory
