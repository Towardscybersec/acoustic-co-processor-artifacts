#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waveguide_acoustic.py

Three acoustic FEM simulations using FEniCS (2016.2.0) + mshr:
 1) Attenuation in a single lossy waveguide
 2) Crosstalk isolation vs. gap for two parallel waveguides
 3) Metamaterial spatial FFT to reveal phononic bandgaps

Compatible with Python 3.5 and DOLFIN 2016.2.0
"""

from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────────────────────────────────────
# Physical & material parameters
vs    = 8433.0       # speed of sound in silicon [m/s]
f0    = 1e8         # center frequency [Hz]
omega = 2*np.pi*f0
Q     = 100.0       # resonator quality factor

# Mesh geometry parameters
L     = 1e-3        # waveguide length [m]
H     = 1e-5        # waveguide height [m]

# Derived complex wavenumber components
a = np.pi*f0/(Q*vs)    # attenuation [1/m]
kr = omega/vs           # real wavenumber [1/m]
ki = a/2.0              # imaginary part for loss

# Ensure output directory
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helmholtz solver with loss (complex k) in mixed real/imag form
def solve_helmholtz(mesh, port_marker, freq, amp=1.0):
    """
    Solve (∇² + (kr+ i ki)^2) p = 0 with Dirichlet=amp on port_marker.
    Returns (u_real, u_imag) as Functions.
    """
    # Function spaces
    V = FunctionSpace(mesh, 'CG', 1)
    W = FunctionSpace(mesh, MixedElement([V.ufl_element(), V.ufl_element()]))
    (u_r, u_i) = TrialFunctions(W)
    (v_r, v_i) = TestFunctions(W)

    # Frequency-dependent kr, ki
    omega_loc = 2*np.pi*freq
    kr_loc    = omega_loc/vs
    ki_loc    = (np.pi*freq)/(Q*vs) / 2.0

    # Variational form
    a = (
        dot(grad(u_r), grad(v_r)) - dot(grad(u_i), grad(v_i))
      - (kr_loc**2 - ki_loc**2)*(u_r*v_r - u_i*v_i)
      + 2*kr_loc*ki_loc*(u_r*v_i + u_i*v_r)
    )*dx + (
        dot(grad(u_r), grad(v_i)) + dot(grad(u_i), grad(v_r))
      - (kr_loc**2 - ki_loc**2)*(u_r*v_i + u_i*v_r)
      - 2*kr_loc*ki_loc*(u_r*v_r - u_i*v_i)
    )*dx
    L = Constant(0.0)*v_r*dx + Constant(0.0)*v_i*dx

    # Boundary conditions
    bc_r = DirichletBC(W.sub(0), Constant(amp), port_marker)
    bc_i = DirichletBC(W.sub(1), Constant(0.0), port_marker)
    bcs = [bc_r, bc_i]

    # Solve
    w = Function(W)
    solve(a == L, w, bcs)
    u_r_sol, u_i_sol = w.split()
    return u_r_sol, u_i_sol

# ─────────────────────────────────────────────────────────────────────────────
# 1) Attenuation in a single waveguide
mesh1 = RectangleMesh(Point(0, 0), Point(L, H), 200, 5)

class LeftPort(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)
port1 = LeftPort()

u1_r, u1_i = solve_helmholtz(mesh1, port1, f0, amp=1.0)

# Sample magnitude along mid-height
xs = np.linspace(0, L, 200)
ps = np.array([math.hypot(u1_r(Point(x, H/2)), u1_i(Point(x, H/2))) for x in xs])

plt.figure()
plt.semilogy(xs*1e3, ps, '-o')
plt.xlabel('Distance (mm)')
plt.ylabel('|p|')
plt.title('Waveguide Attenuation (Q={:.0f})'.format(Q))
plt.grid(True)
plt.savefig('results/attenuation.png')
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 2) Crosstalk isolation vs. gap
gaps = [5e-6, 10e-6, 20e-6, 50e-6]
isol = []
for gap in gaps:
    # Domain: two channels stacked vertically
    mesh2 = RectangleMesh(Point(0, 0), Point(L, 2*H + gap), 200, 50)
    # Lower channel left port
    class PortLower(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0) and x[1] <= H + DOLFIN_EPS
    lower = PortLower()
    u2_r, u2_i = solve_helmholtz(mesh2, lower, f0, amp=1.0)
    # Sample at x=L center of each channel
    P1 = math.hypot(u2_r(Point(L, H/2)), u2_i(Point(L, H/2)))
    P2 = math.hypot(u2_r(Point(L, H + gap + H/2)), u2_i(Point(L, H + gap + H/2)))
    isol.append(20.0*np.log10(P2/P1) if P1 > 1e-16 else -np.inf)

plt.figure()
plt.plot([g*1e6 for g in gaps], isol, '-s')
plt.xlabel('Gap (µm)')
plt.ylabel('Isolation (dB)')
plt.title('Crosstalk vs. Gap')
plt.grid(True)
plt.savefig('results/crosstalk.png')
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3) Metamaterial spatial FFT
Lx, Ly = 4e-3, 2e-3
geom = Rectangle(Point(0, -Ly/2), Point(Lx, Ly/2))
# Subtract periodic circular scatterers
times = 6
for i in range(times):
    geom -= Circle(Point(0.5e-3 + i*0.6e-3, 0.0), 0.00025)
mesh3 = generate_mesh(geom, 64)

u3_r, u3_i = solve_helmholtz(mesh3, port1, f0, amp=1.0)
# Sample along y=0 line
xs3 = np.linspace(0, Lx, 512)
vals = np.array([math.hypot(u3_r(Point(x, 0.0)), u3_i(Point(x, 0.0))) for x in xs3])

# FFT
sp = np.abs(np.fft.fftshift(np.fft.fft(vals)))
freqs = np.fft.fftshift(np.fft.fftfreq(len(xs3), d=(xs3[1]-xs3[0])))

plt.figure()
plt.plot(freqs, sp)
plt.xlabel('Spatial freq (1/m)')
plt.ylabel('FFT(|p|)')
plt.title('Metamaterial Phononic Bandgap')
plt.grid(True)
plt.savefig('results/metamaterial_fft.png')
plt.close()

print("Simulations complete. Results in ./results/")
