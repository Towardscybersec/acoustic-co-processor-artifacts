#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waveguide_acoustic_final.py

Acoustic FEM simulations (FEniCS 2016.2.0 + mshr) using an iterative PETSc CG+AMG solver:
 1) Attenuation in a single waveguide
 2) Crosstalk vs. gap with a 4-layer 3D stacked phononic barrier
 3) Metamaterial spatial FFT with bandgap mapping
 4) Transmission vs. frequency sweep

Compatible with Python 3.5 and DOLFIN 2016.2.0
"""
import os
import math
import numpy as np
import matplotlib
# Use non-interactive Agg backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
from dolfin import PETScKrylovSolver, MPI, mpi_comm_world

# Physical constants
vs = 8433.0        # wave speed (m/s)
f0 = 1e8           # source frequency (Hz)
Q = 50.0           # quality factor

# Derived constants
i_theory = math.pi * f0 / (Q * vs)

# Geometry and solver parameters
domain_length       = 1e-3   # waveguide length (m)
waveguide_height    = 1e-5   # waveguide height (m)
abs_strip_thickness = 0.1 * (math.pi * vs / (2*math.pi * f0))
pml_thickness       = 0.4e-3
sigma_max           = 2.0 * i_theory

# Barrier params (reduced for laptop)
layers            = 2
rows_per_layer    = 16
total_columns     = 32
taper_cells       = 8
radius_factor     = 0.50
res_radius_factor = 0.20
tuned_pitch       = vs/(2*f0)

# Mesh resolution per step (coarse for laptop)
mesh_res_atten = 20      # attenuation step mesh
mesh_res_barrier = 15    # crosstalk barrier mesh
mesh_res_fft = 20        # metamaterial FFT mesh
mesh_res_tx = 20         # transmission sweep mesh

# Define left boundary port
class LeftPort(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)
left_port = LeftPort()

# Helmholtz solver with PETSc CG+AMG
def solve_helmholtz(mesh, port_marker, freq,
                    pml_x_offset=0.0,
                    y_abs_regions=None,
                    amp=1.0,
                    vs=vs,
                    Q=Q,
                    sigma_max=sigma_max,
                    pml_thickness=pml_thickness,
                    max_iters=10000):
    V = FunctionSpace(mesh, 'CG', 1)
    W = FunctionSpace(mesh, MixedElement([V.ufl_element(), V.ufl_element()]))
    u_r, u_i = TrialFunctions(W)
    v_r, v_i = TestFunctions(W)

    omega_loc = 2.0 * math.pi * freq
    k_r_loc = omega_loc / vs
    k_i_base = math.pi * freq / (Q * vs)

    coords = mesh.coordinates()
    x_max = np.max(coords[:, 0])
    x0 = x_max - pml_thickness - pml_x_offset
    sigma_x = Expression(
        'x[0] > x0 ? sigma_max*pow((x[0]-x0)/d,3) : 0.0',
        degree=2, x0=x0, d=pml_thickness, sigma_max=sigma_max
    )
    if y_abs_regions:
        y0, y1, d_y = y_abs_regions
        sigma_y = Expression(
            'x[1] < y0 ? sigma_max*pow((y0-x[1])/d_y,3) '
            ': x[1] > y1 ? sigma_max*pow((x[1]-y1)/d_y,3) : 0.0',
            degree=2, y0=y0, y1=y1, d_y=d_y, sigma_max=sigma_max
        )
    else:
        sigma_y = Constant(0.0)
    sigma = sigma_x + sigma_y

    kr = Constant(k_r_loc)
    ki = k_i_base + sigma

    a_r = dot(grad(u_r), grad(v_r)) - dot(grad(u_i), grad(v_i)) \
          -(kr**2 - ki**2)*(u_r*v_r - u_i*v_i) \
          + 2*kr*ki*(u_r*v_i + u_i*v_r)
    a_i = dot(grad(u_r), grad(v_i)) + dot(grad(u_i), grad(v_r)) \
          -(kr**2 - ki**2)*(u_r*v_i + u_i*v_r) \
          - 2*kr*ki*(u_r*v_r - u_i*v_i)
    a = (a_r + a_i)*dx
    L = Constant(0.0)*(v_r + v_i)*dx

    bc = [
        DirichletBC(W.sub(0), Constant(amp), port_marker),
        DirichletBC(W.sub(1), Constant(0.0), port_marker)
    ]

    w = Function(W)
    A, b = assemble_system(a, L, bc)

    solver = PETScKrylovSolver('cg', 'hypre_amg')
    solver.parameters['relative_tolerance'] = 1e-6
    solver.parameters['maximum_iterations'] = max_iters
    try:
        solver.solve(A, w.vector(), b)
    except RuntimeError:
        LUSolver().solve(A, w.vector(), b)

    return w.split()

# Main routine
def main():
    comm = mpi_comm_world()
    rank = MPI.rank(comm)
    if rank == 0:
        print("Theoretical ki = {0:.3e} m^-1, ~{1:.3f} dB/mm".format(
            i_theory, 20*math.log10(math.e)*i_theory*1e-3
        ))
        os.makedirs('results', exist_ok=True)

    # 1) Attenuation
    # Attenuation mesh: ensure at least 2 divisions in y
    ny = max(2, int(mesh_resolution * waveguide_height / domain_length))
    mesh1 = RectangleMesh(mpi_comm_world(), Point(0,0), Point(domain_length, waveguide_height), mesh_res_atten, max(2, int(mesh_res_atten * waveguide_height / domain_length)))
    u1_r, u1_i = solve_helmholtz(mesh1, left_port, f0)
    xs = np.linspace(0, domain_length, 400)
    p = np.array([math.hypot(u1_r(Point(x, waveguide_height/2)), u1_i(Point(x, waveguide_height/2))) for x in xs])
    mask = xs < (domain_length - pml_thickness)
    slope = np.polyfit(xs[mask], np.log(p[mask]), 1)[0]
    ki_sim = -slope
    if rank == 0:
        print("Fitted ki = {0:.3e} m^-1, ~{1:.3f} dB/mm".format(
            ki_sim, 20*math.log10(math.e)*ki_sim*1e-3
        ))
        plt.figure(); plt.semilogy(xs*1e3, p, '-o'); plt.grid(True)
        plt.savefig('results/attenuation.png'); plt.close()

    # 2) Crosstalk
    isol, gaps = [], [5e-6, 10e-6, 20e-6, 50e-6]
    for gap in gaps:
        height = 2*waveguide_height + gap
        geom = Rectangle(Point(0,0), Point(total_columns*tuned_pitch, height))
        geom -= Rectangle(Point(0,0), Point(abs_strip_thickness, height))
        geom -= Rectangle(Point(total_columns*tuned_pitch-abs_strip_thickness,0), Point(total_columns*tuned_pitch, height))
        for layer in range(layers):
            y_off = layer*(tuned_pitch/2)
            for i in range(total_columns):
                scale = (float(i+1)/taper_cells if i < taper_cells else
                         float(total_columns-i)/taper_cells if i >= total_columns-taper_cells else 1.0)
                rf = radius_factor*tuned_pitch*scale
                rr = res_radius_factor*tuned_pitch
                if rf<=0: continue
                x = tuned_pitch/2 + i*tuned_pitch
                for j in range(rows_per_layer):
                    y = waveguide_height + gap/2 + (j-(rows_per_layer-1)/2)*tuned_pitch + y_off
                    geom -= Circle(Point(x,y), rf)
                    geom += Circle(Point(x,y), rr)
        mesh2 = generate_mesh(geom, mesh_res_barrier)
        u2_r, u2_i = solve_helmholtz(mesh2, left_port, f0, pml_x_offset=tuned_pitch,
                                     y_abs_regions=(waveguide_height, waveguide_height+gap, abs_strip_thickness))
        V2 = FunctionSpace(mesh2,'CG',1)
        bnd = MeshFunction('size_t', mesh2, mesh2.topology().dim()-1); bnd.set_all(0)
        class Out1(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], total_columns*tuned_pitch) and x[1] <= waveguide_height+gap/2
        out1 = Out1()
        class Out2(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], total_columns*tuned_pitch) and x[1] >= waveguide_height+gap/2
        out2 = Out2()
        out1.mark(bnd,1); out2.mark(bnd,2)
        ds = Measure('ds', domain=mesh2, subdomain_data=bnd)
        p_abs = project(sqrt(u2_r**2+u2_i**2),V2)
        P1,P2 = assemble(p_abs*ds(1)), assemble(p_abs*ds(2))
        isol.append(20*math.log10(P2/P1) if P1>0 else -np.inf)
    if rank==0:
        plt.figure(); plt.plot([g*1e6 for g in gaps], isol, '-o'); plt.grid(True)
        plt.savefig('results/crosstalk.png'); plt.close()

    # 3) Metamaterial FFT
    Lx,Ly = 6e-3,2e-3
    geom3 = Rectangle(Point(0,-Ly/2), Point(Lx,Ly/2))
    for i in range(int(Lx/tuned_pitch)):
        geom3 -= Circle(Point(tuned_pitch/2+i*tuned_pitch,0), radius_factor*tuned_pitch)
    mesh3 = generate_mesh(geom3, mesh_res_fft)
    u3_r,u3_i = solve_helmholtz(mesh3,left_port,f0)
    xs3 = np.linspace(0,Lx,4096)
    vals = np.array([math.hypot(u3_r(Point(x,0)),u3_i(Point(x,0))) for x in xs3])
    sp = np.abs(np.fft.fftshift(np.fft.fft(vals*np.hanning(len(vals)),4096)))
    freqs = np.fft.fftshift(np.fft.fftfreq(4096, d=(xs3[1]-xs3[0])))
    if rank==0:
        plt.figure(); plt.plot(freqs, sp);
        # plot theoretical k lines
        plt.axvline(float(math.pi*f0/vs), linestyle='--'); plt.axvline(-float(math.pi*f0/vs), linestyle='--')
        plt.grid(True); plt.savefig('results/metamaterial_fft.png'); plt.close()
        peak,thr = sp.max(),sp.max()/math.sqrt(2)
        mask = sp<thr
        if mask.any(): bw=(freqs[mask].max()-freqs[mask].min()); df=bw*vs/(2*math.pi)
        print("Bandgap Δk={0:.3e} 1/m, Δf≈{1:.3f} MHz".format(bw,df*1e-6))

    # 4) Transmission sweep
    freqs_sweep = np.linspace(f0*0.8,f0*1.2,41)
    trans=[]
    for fr in freqs_sweep:
        geom4=Rectangle(Point(0,0),Point(total_columns*tuned_pitch,2*waveguide_height+20e-6))
        for i in range(total_columns): geom4-=Circle(Point(tuned_pitch/2+i*tuned_pitch,waveguide_height+10e-6),radius_factor*tuned_pitch)
        mesh4 = generate_mesh(geom4, mesh_res_tx)
        p4_r,p4_i=solve_helmholtz(mesh4,left_port,fr)
        V4=FunctionSpace(mesh4,'CG',1)
        b4=MeshFunction('size_t',mesh4,mesh4.topology().dim()-1);b4.set_all(0)
        class Out4(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], total_columns*tuned_pitch)
        out4 = Out4()
        out4.mark(b4, 1)
        ds4=Measure('ds',domain=mesh4,subdomain_data=b4)
        p4=project(sqrt(p4_r**2+p4_i**2),V4)
        trans.append(assemble(p4*ds4(1)))
    if rank==0:
        plt.figure(); plt.plot(freqs_sweep,20*np.log10(np.array(trans)/trans[0])); plt.grid(True)
        plt.savefig('results/transmission_sweep.png'); plt.close()
        print("All simulations complete. Results saved in ./results/!")

if __name__ == '__main__':
    main()
