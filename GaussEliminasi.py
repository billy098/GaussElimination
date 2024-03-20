import numpy as np

def eliminasi_gauss(A, b):
    n = len(A)
    # Menggabungkan matriks A dan vektor b
    Ab = np.column_stack((A, b))

    # Eliminasi maju (forward elimination)
    for k in range(n-1):
        # Pivoting parsial
        indeks_max = np.argmax(np.abs(Ab[k:, k])) + k
        Ab[[k, indeks_max]] = Ab[[indeks_max, k]]

        for i in range(k+1, n):
            faktor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= faktor * Ab[k, k:]

    # Substitusi mundur (back substitution)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, :-1], x)) / Ab[i, i]

    return x

# Contoh penggunaan
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

solusi = eliminasi_gauss(A, b)
print("Solusi:", solusi)
