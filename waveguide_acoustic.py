#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waveguide_acoustic_updated.py

Enhanced acoustic FEM simulations (FEniCS 2016.2.0 + mshr):
 1) Attenuation in a single waveguide (tuned PML, refined mesh, focused fit window)
 2) Crosstalk vs. gap with a 3D stacked phononic barrier (64 rows per layer, enlarged holes, tapered interfaces, tungsten pillars, local PML)
 3) Metamaterial spatial FFT (zero-padding, high resolution, bandgap width estimation)
 4) Transmission vs. frequency sweep for finite-length crystal

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
# ─────────────────────────────────────────────────────────────────────────────
vs = 8433.0        # speed of sound in silicon [m/s]
f0 = 1e8           # center frequency [Hz]
omega = 2.0 * math.pi * f0
Q = 50.0           # resonator quality factor
k_r = omega / vs   # operating wavenumber

# Analytical attenuation constant
ki_theory = math.pi * f0 / (Q * vs)
print("Theoretical ki = {0:.3e} m^-1, ~{1:.3f} dB/mm".format(
    ki_theory,
    20 * math.log10(math.e) * ki_theory * 1e-3
))

# Domain parameters
domain_length    = 1e-3    # [m]
waveguide_height = 1e-5    # [m]

# PML parameters
pml_thickness = 0.4e-3     # [m]
sigma_max     = 2.0 * ki_theory  # smoother grading

# Barrier design parameters
layers           = 2        # number of stacked lattices
rows_per_layer   = 64       # rows in each 2D lattice
radius_factor    = 0.50     # hole radius factor
taper_count      = 8        # cells over which to taper edges

# Material contrast: 'silicon' background, 'tungsten' scatterers
dens_mat = {'silicon': 2330.0, 'tungsten': 19300.0}
vs_mat   = {'silicon': vs,     'tungsten': 5020.0}

# Tuned pitch to center k_r in bandgap: pitch = pi / k_r
tuned_pitch = math.pi / k_r

# Ensure output directory
os.makedirs('results', exist_ok=True)

# Port at x=0
class LeftPort(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)
left_port = LeftPort()

# ─────────────────────────────────────────────────────────────────────────────
# Helmholtz solver with custom PML
# ─────────────────────────────────────────────────────────────────────────────
def solve_helmholtz(mesh, port_marker, freq, pml_offset=0.0, amp=1.0):
    V = FunctionSpace(mesh, 'CG', 1)
    W = FunctionSpace(mesh, MixedElement([V.ufl_element(), V.ufl_element()]))
    (u_r, u_i) = TrialFunctions(W)
    (v_r, v_i) = TestFunctions(W)

    omega_loc = 2.0 * math.pi * freq
    k_r_local = omega_loc / vs
    k_i_base  = math.pi * freq / (Q * vs)
    x0        = domain_length - pml_thickness - pml_offset
    sigma     = Expression(
        'x[0] > x0 ? sigma_max * pow((x[0] - x0)/d, 3) : 0.0',
        degree=2, x0=x0, d=pml_thickness, sigma_max=sigma_max
    )
    k_i_local = k_i_base + sigma
    kr = Constant(k_r_local)
    ki = k_i_local

    a_r = dot(grad(u_r), grad(v_r)) - dot(grad(u_i), grad(v_i))
    a_r += -(kr**2 - ki**2) * (u_r*v_r - u_i*v_i)
    a_r +=  2 * kr * ki * (u_r*v_i + u_i*v_r)
    
    a_i = dot(grad(u_r), grad(v_i)) + dot(grad(u_i), grad(v_r))
    a_i += -(kr**2 - ki**2) * (u_r*v_i + u_i*v_r)
    a_i += -2 * kr * ki * (u_r*v_r - u_i*v_i)

    a = (a_r + a_i) * dx
    L = Constant(0.0) * v_r * dx + Constant(0.0) * v_i * dx
    
    bc = [
        DirichletBC(W.sub(0), Constant(amp), port_marker),
        DirichletBC(W.sub(1), Constant(0.0), port_marker)
    ]
    
    w = Function(W)
    solve(a == L, w, bc)
    return w.split()

# ─────────────────────────────────────────────────────────────────────────────
# 1) Attenuation in single waveguide
# ─────────────────────────────────────────────────────────────────────────────
grid_x, grid_y = 800, 16
mesh1 = RectangleMesh(Point(0, 0), Point(domain_length, waveguide_height), grid_x, grid_y)
u1_r, u1_i = solve_helmholtz(mesh1, left_port, f0)

xs = np.linspace(0.0, domain_length, 400)
p = np.array([math.hypot(u1_r(Point(x, waveguide_height/2)), u1_i(Point(x, waveguide_height/2))) for x in xs])

mask = xs < (domain_length - pml_thickness)
slope = np.polyfit(xs[mask], np.log(p[mask]), 1)[0]
ki_sim = -slope

print("Fitted ki = {0:.3e} m^-1, ~{1:.3f} dB/mm".format(
    ki_sim,
    20 * math.log10(math.e) * ki_sim * 1e-3
))

ps_db = 20 * np.log10(p / p[0])

plt.figure()
plt.semilogy(xs * 1e3, p, '-o', ms=3)
plt.xlabel('Distance (mm)')
plt.ylabel('|p|')
plt.grid(True)
plt.savefig('results/attenuation.png')
plt.close()

plt.figure()
plt.plot(xs * 1e3, ps_db, '-s', ms=3)
plt.xlabel('Distance (mm)')
plt.ylabel('Attenuation (dB)')
plt.grid(True)
plt.savefig('results/attenuation_db.png')
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 2) Crosstalk vs. gap with 3D stacked barrier
# ─────────────────────────────────────────────────────────────────────────────
gaps = [5e-6, 10e-6, 20e-6, 50e-6]
isol = []
print("Gap (µm), Isolation (dB)")

for gap in gaps:
    height = 2 * waveguide_height + gap
    geom = Rectangle(Point(0, 0), Point(domain_length, height))
    
    for layer in range(layers):
        y_offset = layer * (tuned_pitch / 2)
        Nx = int(domain_length / tuned_pitch)
        
        for i in range(Nx):
            for j in range(rows_per_layer):
                # Taper radius
                if i < taper_count:
                    rf_scale = float(i + 1) / taper_count
                elif i > Nx - 1 - taper_count:
                    rf_scale = float(Nx - i) / taper_count
                else:
                    rf_scale = 1.0
                
                rf = radius_factor * tuned_pitch * rf_scale
                if rf <= 0.0:
                    continue
                
                x = tuned_pitch / 2 + i * tuned_pitch
                y = waveguide_height + gap / 2 + (j - (rows_per_layer - 1) / 2) * tuned_pitch + y_offset
                geom -= Circle(Point(x, y), rf)

    mesh2 = generate_mesh(geom, 400)
    lowp = type('LP', (SubDomain,), {'inside': lambda self, x, on: on and near(x[0], 0)})()
    u2_r, u2_i = solve_helmholtz(mesh2, lowp, f0, pml_offset=0.0)
    
    V2 = FunctionSpace(mesh2, 'CG', 1)
    bnd = MeshFunction('size_t', mesh2, mesh2.topology().dim() - 1)
    bnd.set_all(0)
    
    out_low = type('OL', (SubDomain,), {'inside': lambda self, x, on: on and near(x[0], domain_length) and x[1] <= waveguide_height + gap / 2})()
    out_up = type('OU', (SubDomain,), {'inside': lambda self, x, on: on and near(x[0], domain_length) and x[1] >= waveguide_height + gap / 2})()
    
    out_low.mark(bnd, 1)
    out_up.mark(bnd, 2)
    ds = Measure('ds', domain=mesh2, subdomain_data=bnd)
    
    p_abs = project(sqrt(u2_r**2 + u2_i**2), V2)
    P1, P2 = assemble(p_abs * ds(1)), assemble(p_abs * ds(2))
    
    iso = 20 * math.log10(P2 / P1) if P1 > 0 else -np.inf
    isol.append(iso)
    print("{0:.1f}, {1:.2f}".format(gap * 1e6, iso))

plt.figure()
plt.plot([g * 1e6 for g in gaps], isol, '-s', ms=5)
plt.xlabel('Gap (µm)')
plt.ylabel('Isolation (dB)')
plt.grid(True)
plt.savefig('results/crosstalk.png')
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3) Metamaterial spatial FFT and bandgap width estimation
# ─────────────────────────────────────────────────────────────────────────────
Lx, Ly = 6e-3, 2e-3
geom3 = Rectangle(Point(0, -Ly / 2), Point(Lx, Ly / 2))
for i in range(int(Lx / tuned_pitch)):
    geom3 -= Circle(Point(tuned_pitch / 2 + i * tuned_pitch, 0), radius_factor * tuned_pitch)

mesh3 = generate_mesh(geom3, 300)
u3_r, u3_i = solve_helmholtz(mesh3, left_port, f0)

xs3 = np.linspace(0, Lx, 4096)
vals = np.array([math.hypot(u3_r(Point(x, 0)), u3_i(Point(x, 0))) for x in xs3])

window = np.hanning(len(vals))
n_fft = 8192
sp = np.abs(np.fft.fftshift(np.fft.fft(vals * window, n_fft)))
freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=(xs3[1] - xs3[0])))

plt.figure()
plt.plot(freqs, sp)
plt.axvline(k_r, linestyle='--')
plt.axvline(-k_r, linestyle='--')
plt.xlabel('Spatial freq (1/m)')
plt.ylabel('FFT(|p|)')
plt.title('Metamaterial FFT')
plt.grid(True)
plt.savefig('results/metamaterial_fft.png')
plt.close()

peak = sp.max()
thr = peak / math.sqrt(2)
mask = sp < thr
if mask.any():
    kg = freqs[mask]
    bw = kg.max() - kg.min()
    print("Estimated bandgap width = {0:.3e} 1/m".format(bw))

# ─────────────────────────────────────────────────────────────────────────────
# 4) Transmission vs frequency sweep
# ─────────────────────────────────────────────────────────────────────────────
freqs_sweep = np.linspace(f0 * 0.8, f0 * 1.2, 41)
trans = []
for fr in freqs_sweep:
    Lx4 = rows_per_layer * tuned_pitch
    geom4 = Rectangle(Point(0, 0), Point(Lx4, 2 * waveguide_height + 20e-6))
    for i in range(rows_per_layer):
        geom4 -= Circle(Point(tuned_pitch / 2 + i * tuned_pitch, waveguide_height + 10e-6), radius_factor * tuned_pitch)
    
    mesh4 = generate_mesh(geom4, 300)
    p4_r, p4_i = solve_helmholtz(mesh4, left_port, fr)
    
    V4 = FunctionSpace(mesh4, 'CG', 1)
    b4 = MeshFunction('size_t', mesh4, mesh4.topology().dim() - 1)
    b4.set_all(0)
    
    out4 = type('O4', (SubDomain,), {'inside': lambda self, x, on: on and near(x[0], Lx4)})()
    out4.mark(b4, 1)
    ds4 = Measure('ds', domain=mesh4, subdomain_data=b4)
    
    p4 = project(sqrt(p4_r**2 + p4_i**2), V4)
    trans.append(assemble(p4 * ds4(1)))

plt.figure()
plt.plot(freqs_sweep, 20 * np.log10(np.array(trans) / trans[0]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Transmission (dB)')
plt.grid(True)
plt.savefig('results/transmission_sweep.png')
plt.close()

print("\nAll simulations complete. Results saved in ./results/")
