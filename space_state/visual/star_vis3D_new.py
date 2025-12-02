# visual/star_vis3D.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

from base.star_structures import PropagationResult
from base.star_utils import au_to_km
from base.star_constants import BODIES, ts, epoch_jd


class Orbit3D_NEW:
    """
    Minimal, beautiful 3D visualization of a PropagationResult.
    Shows Earth as a solid blue sphere with spacecraft trajectory.
    """
    def __init__(self, result: PropagationResult, title: str = "Orbit 3D"):
        self.result = result
        self.title = title
        self.r_earth_km = au_to_km(BODIES['earth']['radius'])
        self.r_moon_km = au_to_km(BODIES['moon']['radius'])

        # Convert spacecraft position from AU (SSB) → km relative to Earth center
        earth_pos_au = np.array([
            BODIES['earth']['skyfield'].at(ts.tt_jd(epoch_jd + t)).position.au
            for t in result.t
        ]).T  # shape (3, N)

        moon_pos_au = np.array([
            BODIES['moon']['skyfield'].at(ts.tt_jd(epoch_jd + t)).position.au
            for t in result.t
        ]).T  # shape (3, N)

        r_ssb_au = result.y[:3, :]                          # (3, N)
        self.r_rel_km = au_to_km(r_ssb_au - earth_pos_au)    # Earth-centered S/C
        self.moon_rel_km = au_to_km(moon_pos_au - earth_pos_au)  # Earth-centered Moon

    def _earth_sphere(self, ax):
        """Draw solid blue Earth sphere."""
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 60)

        x = self.r_earth_km * np.outer(np.cos(u), np.sin(v))
        y = self.r_earth_km * np.outer(np.sin(u), np.sin(v))
        z = self.r_earth_km * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(x, y, z, color='#1E90FF', alpha=1.0, shade=True, zorder=0)

    def _moon_sphere(self, ax, moon_pos_km):
        """Draw solid gray Moon sphere at position."""
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 30)

        # Translate sphere to moon_pos_km
        x = moon_pos_km[0] + self.r_moon_km * np.outer(np.cos(u), np.sin(v))
        y = moon_pos_km[1] + self.r_moon_km * np.outer(np.sin(u), np.sin(v))
        z = moon_pos_km[2] + self.r_moon_km * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(x, y, z, color='#C0C0C0', alpha=0.8, shade=True, zorder=1)

    def plot(self, show_start_end: bool = True, trail_alpha: float = 0.8):
        """Create the 3D plot with time slider."""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')

        # Earth (static)
        self._earth_sphere(ax)

        # Full trajectory (static, faint)
        x, y, z = self.r_rel_km
        ax.plot(x, y, z, color='#00FFFF', linewidth=1.5, alpha=trail_alpha/2, label='Full Trajectory')

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
        ax.grid(False)
        ax.axis('off')
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False

        # Time slider
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
        slider = Slider(ax_slider, 'Time (days)', 0, len(self.result.t) - 1, valinit=0, valfmt='%d')

        # Initial markers
        self.sc_marker = ax.plot([x[0]], [y[0]], [z[0]], 'o', color='lime', markersize=8)[0]  # [0] for Line3D
        moon_pos_init = self.moon_rel_km[:, 0]
        self.moon_surf = self._add_moon_surface(ax, moon_pos_init)  # New helper for surface
        self.moon_marker = ax.scatter([moon_pos_init[0]], [moon_pos_init[1]], [moon_pos_init[2]], 
                                    color='gray', s=100, label='Moon')

        def update(val):
            idx = int(slider.val)
            t_day = self.result.t[idx]

            # Update S/C marker
            self.sc_marker.remove()
            self.sc_marker = ax.plot([x[idx]], [y[idx]], [z[idx]], 'o', color='lime', markersize=8)[0]

            # Update Moon
            moon_pos_km = self.moon_rel_km[:, idx]
            # Remove old surface (if stored as list of artists)
            if hasattr(self, 'moon_surf_artists'):
                for artist in self.moon_surf_artists:
                    artist.remove()
            self.moon_surf_artists = self._add_moon_surface(ax, moon_pos_km)
            self.moon_marker.remove()
            self.moon_marker = ax.scatter([moon_pos_km[0]], [moon_pos_km[1]], [moon_pos_km[2]], 
                                        color='gray', s=100, label='Moon' if idx == 0 else "")

            # Update title with time
            ax.set_title(f"{self.title} (t = {t_day:.3f} days)", color='white', fontsize=16, pad=30)

            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.tight_layout()
        plt.show()

    def _add_moon_surface(self, ax, moon_pos_km):
        """Helper: Add Moon surface artists (list for removal)."""
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 30)
        x = moon_pos_km[0] + self.r_moon_km * np.outer(np.cos(u), np.sin(v))
        y = moon_pos_km[1] + self.r_moon_km * np.outer(np.sin(u), np.sin(v))
        z = moon_pos_km[2] + self.r_moon_km * np.outer(np.ones_like(u), np.cos(v))
        surf = ax.plot_surface(x, y, z, color='#C0C0C0', alpha=0.8, shade=True, zorder=1)
        return [surf]  # List for batched removal