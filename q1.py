import numpy as np
import matplotlib.pyplot as plt


K = 0.49
dx = 10
dy = 10

nodes = [(i, j) for j in range(4) for i in range(4)]

index = {p: k for k, p in enumerate(nodes)}

A = np.zeros((16, 16))
b = np.zeros(16)

def boundary(i, j):
    if j == 4:
        return 25 * i
    if i == 4:
        return 25 * j


for k, (i, j) in enumerate(nodes):

    
    A[k, k] = 4

   
    if i == 0:
        
        A[k, index[(1, j)]] -= 2
    else:
        for ni in [i-1, i+1]:

            if ni < 4:
                A[k, index[(ni, j)]] -= 1
            else:
                b[k] += boundary(ni, j)

    
    if j == 0:
       
        A[k, index[(i, 1)]] -= 2
    else:
        for nj in [j-1, j+1]:

            if nj < 4:
                A[k, index[(i, nj)]] -= 1
            else:
                b[k] += boundary(i, nj)


def gaussian_elimination(A, b):

    n = len(b)

    M = np.column_stack((A.copy(), b.copy()))

    for i in range(n):

        
        p = i + np.argmax(np.abs(M[i:, i]))
        M[[i, p]] = M[[p, i]]

        for j in range(i+1, n):

            factor = M[j, i] / M[i, i]

            M[j, i:] -= factor * M[i, i:]

    T = np.zeros(n)

    for i in range(n-1, -1, -1):

        T[i] = (
            M[i, -1] -
            np.dot(M[i, i+1:n], T[i+1:n])
        ) / M[i, i]

    return T


T_unknown = gaussian_elimination(A, b)


print("Coefficient Matrix A:")
print(A)

print("\nRight Hand Side Matrix b:")
print(b)

print("\nUnknown Temperatures:")
print(np.round(T_unknown, 4))


T = np.zeros((5, 5))

for j in range(5):
    for i in range(5):

        if i == 4 or j == 4:
            T[j, i] = boundary(i, j)

        else:
            T[j, i] = T_unknown[index[(i, j)]]

print("\nComplete Temperature Distribution (°C):")
print(np.round(T, 4))


x = np.arange(0, 41, 10)
y = np.arange(0, 41, 10)

X, Y = np.meshgrid(x, y)

plt.figure(figsize=(7, 6))

contour = plt.contourf(
    X, Y, T, levels=20, cmap='hot'
)

plt.colorbar(contour, label='Temperature (°C)')

plt.contour(
    X, Y, T, levels=10, colors='black'
)

plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.title('Temperature Distribution in the Plate')

plt.show()


dTdy, dTdx = np.gradient(T, dy, dx)

qx = -K * dTdx
qy = -K * dTdy

qx[:, 0] = 0
qy[0, :] = 0

q = np.sqrt(qx**2 + qy**2)

print("\nHeat Flux in X Direction (qx):")
print(np.round(qx, 4))

print("\nHeat Flux in Y Direction (qy):")
print(np.round(qy, 4))

print("\nHeat Flux Magnitude:")
print(np.round(q, 4))


plt.figure(figsize=(7, 6))

plt.quiver(
    X, Y, qx, qy,
    scale=0.5,
    angles='xy'
)

plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.title('Heat Flux Vector Distribution')

plt.grid()
plt.show()

print("\nQ1 COMPLETED SUCCESSFULLY")