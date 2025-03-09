''' 
Librerías a instalar: pip install numpy matplotlib scipy pandas tqdm
    NumPy (numpy): Generar arreglos aleatorios.
    Matplotlib (matplotlib): Crear gráficos.
    SciPy (scipy.optimize): Ajustar la ecuación cuadrática con curve_fit().
    Pandas (pandas): Manejar los datos de ejecución para graficar.
    tqdm (tqdm): Mostrar barra de progreso.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from time import perf_counter
from tqdm import tqdm  # Importar barra de progreso

# algoritmo suma de sumas
def sums_sum(array):
    sum_s = 0
    for m in range(len(array)): # se itera por el arreglo con m
        sum_m = 0
        for k in range(m + 1):
            sum_m += array[k] # se suman los elementos en el arreglo de k a m + 1
        sum_s += sum_m # se suma el valor de sum_m en cada iteracion a sum_s
    return sum_s # se devuelve la suma de sumas

# datos experimentales
SIZES = [100 * (2 ** i) for i in range(8)]  # 100, 200, 400, ..., 12800
exec_times = [None] * len(SIZES)  # Lista para guardar tiempos

# barra de progreso
with tqdm(total=len(SIZES), desc="Ejecutando SUMS-SUM", unit="iter") as pbar:
    for i, n in enumerate(SIZES):  # por cada valor de los datos experimentales
        start = perf_counter()
        array = np.random.randint(-10, 10, n)  # generar arreglo aleatorio con n de distintos tamaños
        sums_sum(array)  # ejecutar el algoritmo
        end = perf_counter()
        exec_times[i] = end - start  # Guardar tiempo de ejecución
        pbar.update(1)  # Actualizar la barra de progreso

# modelo de la ecuación cuadrática
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

# ajustar ecuación cuadrática a los datos experimentales
opt_params, _ = curve_fit(quadratic_model, SIZES, exec_times)
a, b, c = opt_params

# imprimir tiempos y valores de los datos experimentales
print("\nValores obtenidos (n, t):")
for i in range(len(SIZES)):
    print(f"n = {SIZES[i]:5}, t = {exec_times[i]:.6f} s")

# valores para graficar los datos experimentales
df = pd.DataFrame({"n": SIZES, "t": exec_times})

# valores para graficar t fit
df_fit = pd.DataFrame({
    "n": range(50_000),
    "t_fit": [quadratic_model(x, *opt_params) for x in range(50_000)]
})

# gráfica
ax = df.plot.scatter("n", "t", label="Datos experimentales (sums_sum)")
df_fit.plot(ax=ax, x="n", y="t_fit", ls="--", color="red", label="Ajuste cuadrático (t_fit)")

plt.title("Tiempo de ejecución vs. Tamaño del problema")
plt.xlabel("Tamaño del arreglo (n)")
plt.ylabel("Tiempo de ejecución (segundos)")
plt.grid()
plt.show()
