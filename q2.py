import numpy as np
import matplotlib.pyplot as plt


K = 0.49
C = 0.2174
rho = 2.7

alpha = K / (rho * C)

print("Thermal diffusivity =", alpha, "cm^2/s")


L = 10
dx = 2
dt = 0.1

total_time = 30

x = np.arange(0, L + dx, dx)
n = len(x)

steps = int(total_time / dt)

required_times = [6, 12, 18, 24, 30]

g_left = 1.0

g_right = 0.0


D = np.zeros((n, n))

D[0, 0] = -2 / dx**2
D[0, 1] = 2 / dx**2

for i in range(1, n-1):

    D[i, i-1] = 1 / dx**2
    D[i, i] = -2 / dx**2
    D[i, i+1] = 1 / dx**2

D[-1, -2] = 2 / dx**2
D[-1, -1] = -2 / dx**2


c = np.zeros(n)

c[0] = -2 * g_left / dx
c[-1] = 2 * g_right / dx


I = np.eye(n)

A = I - (alpha * dt / 2) * D

B = I + (alpha * dt / 2) * D

T = np.ones(n) * 50

results = {}

r = alpha * dt / dx**2

print("\nValue of r =", r)


for step in range(1, steps + 1):

    rhs = B @ T + alpha * dt * c

    T = np.linalg.solve(A, rhs)

    current_time = step * dt

    for t in required_times:

        if abs(current_time - t) < 1e-8:

            results[t] = T.copy()


print("\nTEMPERATURE DISTRIBUTION")

print("\nDistance (cm):")
print(x)

for t in required_times:

    print("\nTime =", t, "seconds")

    print("Temperature (degree C):")

    print(np.round(results[t], 4))


plt.figure(figsize=(9, 6))

for t in required_times:

    plt.plot(
        x,
        results[t],
        marker='o',
        label='t = ' + str(t) + ' s'
    )

plt.xlabel("Distance x (cm)")
plt.ylabel("Temperature (degree C)")

plt.title("Temperature Distribution in the Rod")

plt.legend()
plt.grid()

plt.show()


print("\nQ2 COMPLETED SUCCESSFULLY")