#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waveguide_acoustic.py

Three acoustic FEM simulations using FEniCS (2016.2.0) + mshr:
 1) Attenuation in a single lossy waveguide
 2) Crosstalk isolation vs. gap for two parallel waveguides with a reinforced multi-row phononic barrier
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
# Physical & material parameters (all SI units)
vs    = 8433.0       # speed of sound in silicon [m/s]
f0    = 1e8          # center frequency [Hz]
omega = 2.0 * math.pi * f0
Q     = 50.0         # resonator quality factor (lower Q → stronger attenuation)

# Analytical attenuation constant for comparison
ki_theory = math.pi * f0 / (Q * vs)
print("Theoretical ki = {} m^-1, ~{} dB/mm".format(
    ki_theory,
    20.0 * math.log10(math.e) * ki_theory * 1e-3
))

# Note: Implementing PMLs or absorbing boundary conditions at x=L is recommended
# for more accurate attenuation and crosstalk extraction (not implemented here).

# Waveguide geometry parameters
L     = 1e-3         # waveguide length [m]
H     = 1e-5         # waveguide height [m]

# Ensure output directory
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Port subdomain at x=0
class LeftPort(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

left_port = LeftPort()

# ─────────────────────────────────────────────────────────────────────────────
def solve_helmholtz(mesh, port_marker, freq, amp=1.0):
    """
    Solve (∇² + (kr + i·ki)²) u = 0
    Dirichlet BC: u_real = amp, u_imag = 0 on port_marker
    Returns (u_real, u_imag)
    """
    V = FunctionSpace(mesh, 'CG', 1)
    W = FunctionSpace(mesh, MixedElement([V.ufl_element(), V.ufl_element()]))
    (u_r, u_i) = TrialFunctions(W)
    (v_r, v_i) = TestFunctions(W)

    # Local wavenumbers
    omega_loc = 2.0 * math.pi * freq
    kr = omega_loc / vs
    # Corrected attenuation constant (remove /2.0)
    ki = (math.pi * freq) / (Q * vs)

    # Real/imag split bilinear form
    a_real = dot(grad(u_r), grad(v_r)) - dot(grad(u_i), grad(v_i))
    a_real += - (kr**2 - ki**2) * (u_r*v_r - u_i*v_i)
    a_real +=   2 * kr * ki * (u_r*v_i + u_i*v_r)

    a_imag = dot(grad(u_r), grad(v_i)) + dot(grad(u_i), grad(v_r))
    a_imag += - (kr**2 - ki**2) * (u_r*v_i + u_i*v_r)
    a_imag += - 2 * kr * ki * (u_r*v_r - u_i*v_i)

    a_form = (a_real + a_imag) * dx
    L_form = Constant(0.0)*v_r*dx + Constant(0.0)*v_i*dx

    bc_r = DirichletBC(W.sub(0), Constant(amp), port_marker)
    bc_i = DirichletBC(W.sub(1), Constant(0.0), port_marker)

    w = Function(W)
    solve(a_form == L_form, w, [bc_r, bc_i])
    u_r_sol, u_i_sol = w.split()
    return u_r_sol, u_i_sol

# ─────────────────────────────────────────────────────────────────────────────
# 1) Attenuation in a single waveguide
mesh1 = RectangleMesh(Point(0.0, 0.0), Point(L, H), 300, 6)
u1_r, u1_i = solve_helmholtz(mesh1, left_port, f0)

# Sample along centerline
num_pts = 300
xs = np.linspace(0.0, L, num_pts)
ps = [math.hypot(u1_r(Point(x, H/2.0)), u1_i(Point(x, H/2.0))) for x in xs]
ps = np.array(ps)

# Fit exponential decay: ln|p| vs x
ln_ps = np.log(ps)
coeffs = np.polyfit(xs, ln_ps, 1)
slope = coeffs[0]
ki_sim = -slope
print("Fitted ki = {} m^-1, ~{} dB/mm".format(
    ki_sim,
    20.0 * math.log10(math.e) * ki_sim * 1e-3
))

# Convert to dB relative to x=0
ps_db = 20.0 * np.log10(ps/ps[0])

# Plot amplitude (semilog)
plt.figure()
plt.semilogy(xs*1e3, ps, '-', marker='o', markersize=3)
plt.xlabel('Distance (mm)')
plt.ylabel('|p|')
plt.title('Waveguide Attenuation (Q={0:.0f})'.format(Q))
plt.grid(True)
plt.savefig('results/attenuation.png')
plt.close()

# Plot in dB
plt.figure()
plt.plot(xs*1e3, ps_db, '-', marker='s', markersize=3)
plt.xlabel('Distance (mm)')
plt.ylabel('20 log10(|p|/|p(0)|) [dB]')
plt.title('Waveguide Attenuation (dB)')
plt.grid(True)
plt.savefig('results/attenuation_db.png')
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 2) Crosstalk vs. gap with multi-row phononic barrier
gaps = [5e-6, 10e-6, 20e-6, 50e-6]
isol = []
print("Gap(um), Isolation(dB)")

for gap in gaps:
    # Build geometry: two waveguides separated by gap + rows of scatterers
    rows   = 5
    wavelength = vs / f0
    spacing = wavelength / 2.0
    count   = int(L / spacing)
    xs_bar  = np.linspace(spacing/2, L - spacing/2, count)
    radius  = 0.4 * spacing

    geom2 = Rectangle(Point(0.0, 0.0), Point(L, 2*H + gap))
    for r in range(rows):
        y = H + (r+1)*(gap/(rows+1))
        for xb in xs_bar:
            geom2 -= Circle(Point(xb, y), radius)
    mesh2 = generate_mesh(geom2, 200)

    # Define lower-port and upper-port at x=0
    class LowPort(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],0.0) and x[1] <= H + DOLFIN_EPS
    class UpPort(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],0.0) and x[1] >= H+gap - DOLFIN_EPS
    low, up = LowPort(), UpPort()

    u2_r, u2_i = solve_helmholtz(mesh2, low, f0)

    # Improved measurement: integrate pressure over output cross-sections
    V2 = FunctionSpace(mesh2, 'CG', 1)
    boundaries = MeshFunction('size_t', mesh2, mesh2.topology().dim()-1)
    boundaries.set_all(0)
    # Define output line subdomains
    class OutLow(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], L) and x[1] <= H + DOLFIN_EPS
    class OutUp(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], L) and x[1] >= H+gap - DOLFIN_EPS
    out_low, out_up = OutLow(), OutUp()
    out_low.mark(boundaries, 1)
    out_up.mark(boundaries, 2)
    ds = Measure('ds', domain=mesh2, subdomain_data=boundaries)

    # Project absolute pressure
    p_abs_expr = sqrt(u2_r**2 + u2_i**2)
    p_abs = project(p_abs_expr, V2)

    P1 = assemble(p_abs * ds(1))
    P2 = assemble(p_abs * ds(2))
    iso = 20.0 * math.log10(P2 / P1) if P1 > 0 else float('-inf')
    isol.append(iso)
    print("{:.1f}, {:.2f}".format(gap*1e6, iso))

# Plot isolation
gap_um = [g*1e6 for g in gaps]
plt.figure()
plt.plot(gap_um, isol, '-s', markersize=5)
plt.xlabel('Gap (µm)')
plt.ylabel('Isolation (dB)')
plt.title('Crosstalk vs. Gap with Multi-Row Phononic Barrier')
plt.grid(True)
plt.savefig('results/crosstalk.png')
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3) Metamaterial spatial FFT
Lx, Ly = 6e-3, 2e-3
# Updated geometry: use distinct, non-tangent holes
pitch = 0.5e-3  # lattice pitch
radius = 0.24e-3  # hole radius < pitch/2
num_holes = int(Lx / pitch)
geom3 = Rectangle(Point(0.0, -Ly/2), Point(Lx, Ly/2))
for i in range(num_holes):
    geom3 -= Circle(Point(pitch/2 + i*pitch, 0.0), radius)
mesh3 = generate_mesh(geom3, 100)

u3_r, u3_i = solve_helmholtz(mesh3, left_port, f0)
# Dense sampling along centerline
num_fft = 4096
xs3 = np.linspace(0.0, Lx, num_fft)
vals = np.array([math.hypot(u3_r(Point(x,0.0)), u3_i(Point(x,0.0))) for x in xs3])
# Apply window to reduce spectral leakage
window = np.hanning(num_fft)
vals_w = vals * window
# Compute FFT
sp    = np.abs(np.fft.fftshift(np.fft.fft(vals_w)))
freqs = np.fft.fftshift(np.fft.fftfreq(num_fft, d=(xs3[1]-xs3[0])))
# Theoretical Bragg wave-number
k_bragg = math.pi / pitch

plt.figure()
plt.plot(freqs, sp, '-')
plt.axvline(k_bragg, linestyle='--', label='Bragg k')
plt.axvline(-k_bragg, linestyle='--')
plt.xlabel('Spatial freq (1/m)')
plt.ylabel('FFT(|p|)')
plt.title('Metamaterial Phononic Bandgap')
plt.legend()
plt.grid(True)
plt.savefig('results/metamaterial_fft.png')
plt.close()

print("Simulations complete. Results saved in ./results/")
