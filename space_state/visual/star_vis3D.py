# star_vis3D.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from base.star_structures import PropagationResult
from base.star_utils import au_to_km
from base.star_constants import BODIES, ts, epoch_jd


class Orbit3D:
    """
    Minimal, beautiful 3D visualization of a PropagationResult.
    Shows Earth as a solid blue sphere with spacecraft trajectory.
    """
    def __init__(self, result: PropagationResult, title: str = "Orbit 3D"):
        self.result = result
        self.title = title
        self.r_earth_km = au_to_km(BODIES['earth']['radius'])

        # Convert spacecraft position from AU (SSB) → km relative to Earth center
        earth_pos_au = np.array([
            BODIES['earth']['skyfield'].at(ts.tt_jd(epoch_jd + t)).position.au
            for t in result.t
        ]).T  # shape (3, N)

        r_ssb_au = result.y[:3, :]                          # (3, N)
        self.r_rel_km = au_to_km(r_ssb_au - earth_pos_au)    # Earth-centered

    def _earth_sphere(self, ax):
        """Draw solid blue Earth sphere."""
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 60)

        x = self.r_earth_km * np.outer(np.cos(u), np.sin(v))
        y = self.r_earth_km * np.outer(np.sin(u), np.sin(v))
        z = self.r_earth_km * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(x, y, z, color='#1E90FF', alpha=1.0, shade=True, zorder=0)

    def plot(self, show_start_end: bool = True, trail_alpha: float = 0.8):
        """Create the 3D plot."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')

        # Earth
        self._earth_sphere(ax)

        # Spacecraft trajectory
        x, y, z = self.r_rel_km
        ax.plot(x, y, z, color='#00FFFF', linewidth=2.5, alpha=trail_alpha, label='Trajectory')

        # Start and end points
        if show_start_end:
            ax.plot([x[0]], [y[0]], [z[0]], 'o', color='lime', markersize=10, label='Start')
            ax.plot([x[-1]], [y[-1]], [z[-1]], 'o', color='red', markersize=10, label='End')

        # Styling
        ax.set_title(self.title, color='white', fontsize=16, pad=30)
        ax.legend(facecolor='black', labelcolor='white', loc='upper left')

        # Equal aspect + nice view
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(elev=30, azim=45)

        # Clean look
        ax.grid(False)
        ax.axis('off')
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False

        plt.tight_layout()
        plt.show()


# ————————————————————
# Usage (one line!)
# ————————————————————
# from star_vis3D import Orbit3D
# Orbit3D(prop.result, title="Starship → Moon Transfer").plot()