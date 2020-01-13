import matplotlib.pyplot as plt
import math
import numpy as np

def solve_euler_explicite(f, x0, dt, tf, t0 = 0):
    t, x = [t0], [x0]
    while t[-1] < tf:
        x.append(x[-1] + dt * f(t[-1], x[-1]))
        t.append(t[-1] + dt)
    return t, x

def solve_heun(f, x0, dt, tf, t0 = 0):
    t, x = [t0], [x0]
    while t[-1] < tf:
        t.append(t[-1] + dt)
        x.append(x[-1] + dt/2 * (f(t[-2], x[-1]) + f(t[-1], x[-1] + dt * f(t[-2], x[-1])))) 
    return t, x

def erreur_globale(f, g, x0, dt, tf, t0 = 0):
    t, x = solve_euler_explicite(f, x0, dt, tf, t0)
    n = len(t)
    err = 0
    for j in range(n):
        if abs(g(t[j]) - x[j]) > err:
            err = abs(g(t[j]) - x[j])
    return err

def erreur_globale_heun(f, g, x0, dt, tf, t0 = 0):
    t, x = solve_heun(f, x0, dt, tf, t0)
    n = len(t)
    err = 0
    for j in range(n):
        if abs(g(t[j]) - x[j]) > err:
            err = abs(g(t[j]) - x[j])
    return err   


def f(t, x):
    return x
def exp(y):
    return math.exp(y)

pas = np.linspace(0.001, 0.01, 1000)
erreur = np.array([erreur_globale_heun(f, exp, 1, dt, 3) for dt in pas])

#print(erreur_globale(f, exp, 1, 0.1, 10))
plt.plot(pas**2, erreur)
plt.show()


#Consignes pour vérifier que le solver fonctionne
"""t = solve_euler_explicite(f, 1, 0.1, 10)[0]
x = solve_euler_explicite(f, 1, 0.1, 10)[1]
x2 = np.array([exp(y) for y in t])

plt.plot(t, x2, label = "solution Euler explicite")
plt.plot(t, x, label = "solution exacte")
plt.legend()
plt.show()"""












