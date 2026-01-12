import sys
import numpy as np
from ase.io import read, write
from ase.build import make_supercell

# ==========================================
# 1. User Configuration
# ==========================================

# Supercell dimensions [Nx, Ny, Nz]
DIM = [16, 16, 16]

# Number of k-points per segment
NK = 100

# High symmetry path (Fractional Coordinates)
# Default: FCC path (e.g., for MgO) Gamma -> X -> K -> Gamma -> L
K_PATH_POINTS = [
    [0.0, 0.0, 0.0],          # Gamma
    [0.5, 0.0, 0.5],          # X
    [0.375, 0.375, 0.75],     # K
    [0.0, 0.0, 0.0],          # Gamma
    [0.5, 0.5, 0.5]           # L
]

# ==========================================
# 2. Main Logic
# ==========================================

def generate_all(poscar_file):
    """
    Reads POSCAR, generates supercell (model.xyz), basis.in, and kpoints.in.
    """
    print(f"--- Reading Primitive POSCAR: {poscar_file} ---")
    try:
        prim_cell = read(poscar_file, format='vasp')
    except Exception as e:
        print(f"Error reading POSCAR: {e}")
        return

    # -------------------------------------------------
    # Task A: Generate Supercell (model.xyz)
    # -------------------------------------------------
    print(f"--- Generating Supercell {DIM} ---")
    
    # Create diagonal transformation matrix from DIM
    transform_matrix = np.diag(DIM)
    supercell = make_supercell(prim_cell, transform_matrix)
    
    # Save to extended XYZ format for GPUMD
    write("model.xyz", supercell, format='extxyz')
    print("Done: model.xyz")

    # -------------------------------------------------
    # Task B: Generate basis.in
    # -------------------------------------------------
    print(f"--- Generating basis.in ---")
    
    # Get Lattice Info
    prim_lattice = prim_cell.get_cell()
    prim_lattice_inv = np.linalg.inv(prim_lattice)
    
    n_prim = len(prim_cell)
    n_super = len(supercell)
    prim_masses = prim_cell.get_masses()
    prim_positions_frac = prim_cell.get_scaled_positions()
    supercell_positions = supercell.get_positions()
    
    # 1. Calculate Mapping (Supercell Atom -> Primitive Atom ID)
    map_list = []
    
    for idx, pos in enumerate(supercell_positions):
        # Cartesian -> Primitive Fractional -> Wrap to [0, 1)
        frac_coords = np.dot(pos, prim_lattice_inv)
        folded_coords = frac_coords % 1.0
        
        # Find nearest primitive atom
        min_dist = float('inf')
        mapped_id = -1
        
        for j in range(n_prim):
            prim_pos = prim_positions_frac[j]
            diff = folded_coords - prim_pos
            diff -= np.round(diff) # Minimum image convention
            cart_diff = np.dot(diff, prim_lattice)
            dist = np.linalg.norm(cart_diff)
            
            if dist < min_dist:
                min_dist = dist
                mapped_id = j
        
        if min_dist > 0.5:
            print(f"Warning: Large mapping error for atom {idx}: {min_dist:.4f} A")
        
        map_list.append(mapped_id)

    # 2. Find representatives for Basis block
    basis_representatives = []
    for p in range(n_prim):
        try:
            # Find first atom in supercell mapped to primitive atom p
            rep_id = map_list.index(p)
            mass = prim_masses[p]
            basis_representatives.append((rep_id, mass))
        except ValueError:
            print(f"Error: Primitive atom {p} not mapped in supercell.")
            return

    # 3. Write basis.in
    with open("basis.in", 'w') as f:
        f.write(f"{n_prim}\n")
        # Basis definition
        for i in range(n_prim):
            rep_id, mass = basis_representatives[i]
            f.write(f"{rep_id} {mass:.6f}\n")
        # Mapping
        for i in range(n_super):
            f.write(f"{map_list[i]}\n")
            
    print("Done: basis.in")

    # -------------------------------------------------
    # Task C: Generate kpoints.in
    # -------------------------------------------------
    print(f"--- Generating kpoints.in ---")

    # 1. Calculate Reciprocal Lattice
    # FIX: Use atoms.cell.reciprocal() instead of deprecated get_reciprocal_cell()
    # Multiply by 2pi for physics definition (1/Angstrom)
    recip_cell = prim_cell.cell.reciprocal() * 2 * np.pi
    
    # 2. Generate K-point path
    all_k_cart = []
    
    for i in range(len(K_PATH_POINTS) - 1):
        start_frac = np.array(K_PATH_POINTS[i])
        end_frac = np.array(K_PATH_POINTS[i+1])
        
        # Linear interpolation
        frac_points = np.linspace(start_frac, end_frac, NK)
        
        # Convert to Cartesian (1/Angstrom)
        cart_points = np.dot(frac_points, recip_cell)
        all_k_cart.extend(cart_points)
    
    # 3. Write kpoints.in
    with open("kpoints.in", 'w') as f:
        f.write(f"{len(all_k_cart)}\n")
        for k in all_k_cart:
            f.write(f"{k[0]:.8f} {k[1]:.8f} {k[2]:.8f}\n")
            
    print(f"Done: kpoints.in ({len(all_k_cart)} points)")

if __name__ == "__main__":
    primitive_cell = 'POSCAR'
    generate_all(primitive_cell)
