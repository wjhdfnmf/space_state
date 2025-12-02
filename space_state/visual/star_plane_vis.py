import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from base.star_utils import au_to_km

def plot_orbital_plane(r1: np.ndarray, v1: np.ndarray, r2: np.ndarray, v2: np.ndarray):
    """Plot the orbital plane defined by position r and velocity v vectors."""

    # Orbital angular momentum vector (plane normal)
    h = np.cross(r1, v1)
    h_hat = h / np.linalg.norm(h)

    # Orbital angular momentum vector (plane normal)
    k = np.cross(r2, v2)
    k_hat = k / np.linalg.norm(k)

    # Node line = intersection of orbital plane with reference plane
    n = np.cross(k_hat, h_hat)
    n_hat = n / np.linalg.norm(n)

    # Create grid for orbital plane
    plane_size = 9000
    grid_x, grid_y = np.meshgrid(np.linspace(-plane_size, plane_size, 10),
                                np.linspace(-plane_size, plane_size, 10))
    # Orbital plane equation: h_hat ⋅ (x,y,z) = 0 → solve for z
    grid_h = -(h_hat[0] * grid_x + h_hat[1] * grid_y) / h_hat[2]

    grid_k = -(k_hat[0] * grid_x + k_hat[1] * grid_y) / k_hat[2]

    grid_z = np.zeros_like(grid_x)

    # Plot
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot orbital plane
    ax.plot_surface(grid_x, grid_y, grid_h, alpha=0.4, color='blue')

    # Plot reference plane (XY-plane)
    ax.plot_surface(grid_x, grid_y, grid_k, alpha=0.4, color='red')

    # Plot reference plane (XY-plane)
    ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.4, color='gray')

    # Plot node line
    t_vals = np.linspace(-10000, 10000, 100)
    node_line = np.outer(t_vals, n_hat)
    ax.plot(node_line[:,0], node_line[:,1], node_line[:,2], linewidth=3)

    # Plot r vectors
    arrow_len = plane_size * 0.3
    r1_norm = r1 / np.linalg.norm(r1)
    v1_norm = v1 / np.linalg.norm(v1)
    r2_norm = r2 / np.linalg.norm(r2)
    v2_norm = v2 / np.linalg.norm(v2)

    ax.quiver(0, 0, 0, *(r1_norm * arrow_len), color="orange")
    ax.quiver(0, 0, 0, *(v1_norm * arrow_len), color="green")
    #ax.quiver(0, 0, 0, *(r2_norm * arrow_len), color="orange")
    #ax.quiver(0, 0, 0, *(v2_norm * arrow_len), color="green")

    # Labels
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Orbital Plane (blue), Target Plane (red), Reference Plane (gray), and Node Line")

    plt.show()
