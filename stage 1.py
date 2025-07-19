#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waveguide_acoustic.py

Three acoustic FEM simulations using FEniCS + mshr:
 1) Attenuation in a single lossy waveguide
 2) Crosstalk isolation vs. gap for two parallel waveguides
 3) Metamaterial spectral response via spatial FFT

Compatible with Python 3.5 and FEniCS 2016.2.0
"""

from fenics import *
from mshr import Rectangle, Circle, generate_mesh
import numpy as np
import matplotlib.pyplot as plt
import cmath, math, os

# ─────────────────────────────────────────────────────────────────────────────
# Physical & material parameters
c0       = 8433.0       # speed of sound in silicon (m/s)
f        = 1e9          # frequency (Hz)
omega    = 2.0*math.pi*f
k_real   = omega / c0   # real part of wavenumber
alpha    = 1.18e5       # loss coefficient (1/m)
k_imag   = alpha        # imaginary part of wavenumber
a2       = k_real**2 - k_imag**2  # real(k^2)
b2       = 2.0*k_real*k_imag      # imag(k^2)

# Domain parameters
L        = 1e-3         # length (m)
H        = 10e-6        # single waveguide height (m)

# Mesh resolutions
grid_res = 64           # for single waveguide
cross_res = 128         # for crosstalk domain
meta_res  = 64          # for metamaterial domain

# Ensure output folder
os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
def setup_mixed(mesh):
    """Mixed FunctionSpace for complex pressure (real + imag)."""
    P1 = FunctionSpace(mesh, 'Lagrange', 1)
    element = MixedElement([P1.ufl_element(), P1.ufl_element()])
    return FunctionSpace(mesh, element)


def robin_absorb(u_r, u_i, v_r, v_i, ds):
    """Weak form terms for ∂n p = i k p absorbing BC."""
    term = (
        b2*(u_r*v_r + u_i*v_i)
      - k_real*(u_i*v_r - u_r*v_i)
      + k_imag*(u_r*v_r + u_i*v_i)
    )*ds
    return term

# ─────────────────────────────────────────────────────────────────────────────
def solve_single():
    # Generate mesh
    domain = Rectangle(Point(0,0), Point(L,H))
    mesh   = generate_mesh(domain, grid_res)
    V      = setup_mixed(mesh)

    # Trial/test
    u, v    = TrialFunction(V), TestFunction(V)
    u_r, u_i = split(u); v_r, v_i = split(v)

    # Helmholtz weak form
    a_vol = (
        dot(grad(u_r), grad(v_r)) - dot(grad(u_i), grad(v_i))
      - a2*(u_r*v_r - u_i*v_i)
      + dot(grad(u_r), grad(v_i)) + dot(grad(u_i), grad(v_r))
      - a2*(u_r*v_i + u_i*v_r)
    )*dx

    # Absorbing BC on outlet x=L
    facets = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    AutoSubDomain(lambda x,on: on and near(x[0],L)).mark(facets, 1)
    ds = Measure('ds', domain=mesh, subdomain_data=facets)(1)
    a_bc = robin_absorb(u_r, u_i, v_r, v_i, ds)

    # Dirichlet BC at inlet x=0 (p=1+0j)
    inlet = AutoSubDomain(lambda x,on: on and near(x[0],0.0))
    bc_r = DirichletBC(V.sub(0), Constant(1.0), inlet)
    bc_i = DirichletBC(V.sub(1), Constant(0.0), inlet)

    a = a_vol + a_bc
    Lf = Constant(0.0)*v_r*dx + Constant(0.0)*v_i*dx

    sol = Function(V)
    solve(a == Lf, sol, [bc_r, bc_i])
    pr, pi = sol.split()

    # Sample magnitude along centerline
    xs = np.linspace(0, L, 300)
    mags = [math.hypot(pr(x,H/2), pi(x,H/2)) for x in xs]

    plt.figure()
    plt.semilogy(xs*1e3, mags)
    plt.xlabel('x (mm)'); plt.ylabel('|p|')
    plt.title('Attenuation along waveguide')
    plt.grid(True)
    plt.savefig('results/attenuation.png')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
def solve_crosstalk():
    gaps = [2e-6,5e-6,10e-6,20e-6,50e-6]
    isol = []
    for gap in gaps:
        H2 = 2*H + gap
        mesh2 = generate_mesh(Rectangle(Point(0,0), Point(L,H2)), cross_res)
        V2    = setup_mixed(mesh2)
        u, v  = TrialFunction(V2), TestFunction(V2)
        ur, ui = split(u); vr, vi = split(v)
        a_vol = (
            dot(grad(ur),grad(vr)) - dot(grad(ui),grad(vi))
          - a2*(ur*vr - ui*vi)
          + dot(grad(ur),grad(vi)) + dot(grad(ui),grad(vr))
          - a2*(ur*vi + ui*vr)
        )*dx

        # Absorb at x=L
        facets2 = MeshFunction('size_t', mesh2, mesh2.topology().dim()-1, 0)
        AutoSubDomain(lambda x,on: on and near(x[0],L)).mark(facets2,1)
        ds2 = Measure('ds', domain=mesh2, subdomain_data=facets2)(1)
        a_bc2 = robin_absorb(ur, ui, vr, vi, ds2)

        # Inlet BCs: bottom guide p=1, top p=0
        markers = MeshFunction('size_t', mesh2, mesh2.topology().dim()-1, 0)
        AutoSubDomain(lambda x,on: on and near(x[0],0.0) and x[1]<=H).mark(markers,1)
        AutoSubDomain(lambda x,on: on and near(x[0],0.0) and x[1]>=H+gap).mark(markers,2)
        bc1 = DirichletBC(V2.sub(0), Constant(1.0), markers, 1)
        bc2 = DirichletBC(V2.sub(1), Constant(0.0), markers, 1)
        bc3 = DirichletBC(V2.sub(0), Constant(0.0), markers, 2)
        bc4 = DirichletBC(V2.sub(1), Constant(0.0), markers, 2)

        a2_form = a_vol + a_bc2
        Lf2     = Constant(0.0)*vr*dx + Constant(0.0)*vi*dx

        sol2 = Function(V2)
        solve(a2_form == Lf2, sol2, [bc1,bc2,bc3,bc4])
        pr2, pi2 = sol2.split()

        # Measure at top guide outlet (x=L)
        y_out = H + gap/2
        P2    = math.hypot(pr2(L,y_out), pi2(L,y_out))
        isol.append(20.0*math.log10(P2))

    plt.figure()
    plt.plot([g*1e6 for g in gaps], isol, 'o-')
    plt.xlabel('Gap (µm)'); plt.ylabel('Isolation (dB)')
    plt.title('Crosstalk vs. gap')
    plt.grid(True)
    plt.savefig('results/crosstalk.png')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
def solve_metamaterial():
    # Create straight channel mesh (replace with true scatterer geometry as needed)
    mesh3 = generate_mesh(Rectangle(Point(0,0), Point(L,H)), meta_res)
    V3    = setup_mixed(mesh3)
    u, v  = TrialFunction(V3), TestFunction(V3)
    ur, ui = split(u); vr, vi = split(v)
    a_vol = (
        dot(grad(ur),grad(vr)) - dot(grad(ui),grad(vi))
      - a2*(ur*vr - ui*vi)
      + dot(grad(ur),grad(vi)) + dot(grad(ui),grad(vr))
      - a2*(ur*vi + ui*vr)
    )*dx

    # Absorb at x=L
    facets3 = MeshFunction('size_t', mesh3, mesh3.topology().dim()-1, 0)
    AutoSubDomain(lambda x,on: on and near(x[0],L)).mark(facets3,1)
    ds3 = Measure('ds', domain=mesh3, subdomain_data=facets3)(1)
    a_bc3 = robin_absorb(ur, ui, vr, vi, ds3)

    # Inlet p=1 at x=0
    inlet3 = AutoSubDomain(lambda x,on: on and near(x[0],0.0))
    bc3r   = DirichletBC(V3.sub(0), Constant(1.0), inlet3)
    bc3i   = DirichletBC(V3.sub(1), Constant(0.0), inlet3)

    a3 = a_vol + a_bc3
    Lf3 = Constant(0.0)*vr*dx + Constant(0.0)*vi*dx

    sol3 = Function(V3)
    solve(a3 == Lf3, sol3, [bc3r, bc3i])
    pr3, pi3 = sol3.split()

    # Sample output vs y
    Ys = np.linspace(0, H, 300)
    vals = [pr3(L,y) + 1j*pi3(L,y) for y in Ys]
    spec = np.fft.fft(np.abs(vals))
    freq = np.fft.fftfreq(len(Ys), d=Ys[1]-Ys[0])

    plt.figure()
    plt.plot(freq, np.abs(spec))
    plt.xlim(-5e5, 5e5)
    plt.xlabel('spatial freq (1/m)'); plt.ylabel('FFT(|p|)')
    plt.title('Metamaterial spectral response')
    plt.grid(True)
    plt.savefig('results/metamaterial_fft.png')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Single waveguide ===')   ; solve_single()
    print('Saved attenuation.png')      

    print('=== Crosstalk ===')           ; solve_crosstalk()
    print('Saved crosstalk.png')        

    print('=== Metamaterial FFT ===')    ; solve_metamaterial()
    print('Saved metamaterial_fft.png')  

    print('\nAll simulations complete. Results in ./results/')
