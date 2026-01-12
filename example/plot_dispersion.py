import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

def get_k_distances_from_poscar(poscar_file, k_points, points_per_segment):
    """
    读取 POSCAR 计算倒易晶格，并生成 K 点路径的线性距离 (X轴)
    """
    try:
        atoms = read(poscar_file, format='vasp')
    except Exception as e:
        print(f"Error reading {poscar_file}: {e}")
        exit(1)
        
    # 获取倒易晶格矩阵 (乘以 2pi)
    recip_cell = atoms.cell.reciprocal() * 2 * np.pi

    k_dist_flat = []
    ticks = [0.0]
    current_dist = 0.0
    
    for i in range(len(k_points) - 1):
        start_k_frac = np.array(k_points[i])
        end_k_frac = np.array(k_points[i+1])
        
        # 转换分数坐标差值到倒易空间笛卡尔坐标
        delta_frac = end_k_frac - start_k_frac
        delta_cart = np.dot(delta_frac, recip_cell)
        dist_segment = np.linalg.norm(delta_cart)
        
        if i == 0:
            segment_dists = np.linspace(current_dist, current_dist + dist_segment, points_per_segment)
            k_dist_flat.extend(segment_dists)
        else:
            segment_dists = np.linspace(current_dist, current_dist + dist_segment, points_per_segment)
            k_dist_flat.extend(segment_dists[1:])
            
        current_dist += dist_segment
        ticks.append(current_dist)
        
    return np.array(k_dist_flat), np.array(ticks)

# ================= 配置区域 =================

POSCAR_FILE = "POSCAR"       
DATA_FILE = "omega2.out"     

#  高对称点路径
# Gamma -> X -> K -> Gamma -> L
special_k_points = [
    [0.0, 0.0, 0.0],          # Gamma
    [0.5, 0.0, 0.5],          # X
    [0.375, 0.375, 0.75],     # K (3/8, 3/8, 3/4)
    [0.0, 0.0, 0.0],          # Gamma
    [0.5, 0.5, 0.5]           # L
]

labels = [r'$\Gamma$', 'X', 'K', r'$\Gamma$', 'L']

NK = 100 

# ================= 主程序 =================

x, ticks = get_k_distances_from_poscar(POSCAR_FILE, special_k_points, NK)

print(f"Reading {DATA_FILE}...")
try:
    omega2 = np.loadtxt(DATA_FILE)
except IOError:
    print(f"Error: {DATA_FILE} not found.")
    exit()

if omega2.shape[0] != len(x):
    print(f"Warning: Data shape {omega2.shape} vs X-axis {len(x)}.")
    if omega2.shape[1] == len(x):
        omega2 = omega2.T
    elif abs(omega2.shape[0] - len(x)) <= len(special_k_points):
        x = np.linspace(0, x[-1], omega2.shape[0])
        ticks = ticks * (omega2.shape[0] / len(x))

# 转换单位: omega^2 -> THz
freqs = np.sqrt(omega2.astype(complex)) 
freqs = np.real(freqs) / (2 * np.pi) 

plt.figure(figsize=(8, 6))

plt.plot(x, freqs, color='tab:blue', linewidth=1.5)

for tick in ticks:
    plt.axvline(x=tick, color='k', linestyle='-', linewidth=0.5)

plt.xticks(ticks, labels, fontsize=12)
plt.xlim(x[0], x[-1])
plt.ylabel(r'Frequency (THz)', fontsize=12)
plt.ylim(0, np.max(freqs) * 1.1)
plt.title('Phonon Dispersion', fontsize=14)
plt.tight_layout()
plt.savefig('phonon_dispersion.pdf')
plt.show()
print("Done. Saved to phonon_dispersion.pdf")
