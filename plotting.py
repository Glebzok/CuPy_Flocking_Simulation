import cupy as cp
import matplotlib
matplotlib.use('TkAgg') # tested with 'GTK3Agg', 'TkAgg', 'TkCairo', 'WebAgg' backends
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

from simulation import Flock


class FlockPlotter(object):
    def __init__(self, flock):
        self.flock = flock
        self.n_obj = flock.n_obj

        self.sliders_meta = \
            {'n_obj': {'bounds': (1, 2000), 'default_value': 10, 'label': '# Objects'},
             'alpha_alignment': {'bounds': (0, 1), 'default_value': 0.5, 'label': 'Alignment'},
             'alpha_separation': {'bounds': (0, 5), 'default_value': 1, 'label': 'Separation'},
             'alpha_cohesion': {'bounds': (0, 5), 'default_value': 1, 'label': 'Cohesion'},
             'alpha_random': {'bounds': (0, 5), 'default_value': 1, 'label': 'Randomness'},
             'alpha_boundary_avoidance': {'bounds': (0, 50), 'default_value': 20, 'label': 'Boundary Avoidance'},
             'bound_threshold': {'bounds': (0, 50), 'default_value': 13, 'label': 'Boundary Threshold'},
             'r_vision': {'bounds': (0, 10), 'default_value': 3, 'label': 'Vision'},
             'r_personal_space': {'bounds': (0, 5), 'default_value': 1, 'label': 'Private Space'}}

    def update_plot(self, i, scat):
        if self.n_obj > self.flock.n_obj:
            self.flock.add_new_objs(self.n_obj - self.flock.n_obj)
            self.flock.n_obj = self.n_obj
        elif self.n_obj < self.flock.n_obj:
            self.flock.delete_objs(self.flock.n_obj - self.n_obj)
            self.flock.n_obj = self.n_obj

        distances = self.flock.calc_distance_matrix()
        self.flock.update_v(distances)
        self.flock.update_r()

        c = cp.arctan2(self.flock.state[:, 2], self.flock.state[:, 3]) / cp.pi / 2 + 0.5
        scat.set_color(plt.get_cmap('hsv')(c.get()))
        scat.set_offsets(self.flock.state[:, :2].get())
        return scat,

    def update_n_obj(self, val):
        self.n_obj = int(val)

    def update_alpha_alignment(self, val):
        self.flock.alpha_alignment = val

    def update_alpha_separation(self, val):
        self.flock.alpha_separation = val

    def update_alpha_cohesion(self, val):
        self.flock.alpha_cohesion = val

    def update_alpha_random(self, val):
        self.flock.alpha_random = val

    def update_alpha_boundary_avoidance(self, val):
        self.flock.alpha_boundary_avoidance = val

    def update_bound_threshold(self, val):
        self.flock.bound_threshold = val

    def update_r_vision(self, val):
        self.flock.r_vision = val

    def update_r_personal_space(self, val):
        self.flock.r_personal_space = val

    def update_slider(self, name):
        if name == 'n_obj':
            return self.update_n_obj
        elif name == 'alpha_alignment':
            return self.update_alpha_alignment
        elif name == 'alpha_separation':
            return self.update_alpha_separation
        elif name == 'alpha_cohesion':
            return self.update_alpha_cohesion
        elif name == 'alpha_random':
            return self.update_alpha_random
        elif name == 'alpha_boundary_avoidance':
            return self.update_alpha_boundary_avoidance
        elif name == 'bound_threshold':
            return self.update_bound_threshold
        elif name == 'r_vision':
            return self.update_r_vision
        elif name == 'r_personal_space':
            return self.update_r_personal_space
        else:
            raise ValueError(f'Wrong name: {name}')

    def animate(self, n_iter):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle('Flocking simulation')

        numframes = n_iter

        plt.gca().set_facecolor('black')
        plt.xlim((self.flock.x_min, self.flock.x_max))
        plt.ylim((self.flock.y_min, self.flock.y_max))
        plt.xticks([])
        plt.yticks([])

        c = cp.arctan2(self.flock.state[:, 2], self.flock.state[:, 3]) / cp.pi / 2 + 0.5
        scat = plt.scatter(self.flock.state[:, 0].get(), self.flock.state[:, 1].get(), s=10, c=c.get())

        plt.subplots_adjust(left=0.5, bottom=0)
        sliders = []

        for i, slider in enumerate(self.sliders_meta):
            ax = plt.axes([0.23, 0.2 + 0.05 * i, 0.2, 0.03])
            slider_widget = Slider(ax=ax,
                                   label=self.sliders_meta[slider]['label'],
                                   valmin=self.sliders_meta[slider]['bounds'][0],
                                   valmax=self.sliders_meta[slider]['bounds'][1],
                                   valinit=self.sliders_meta[slider]['default_value'])

            slider_widget.on_changed(self.update_slider(slider))
            sliders.append(slider_widget)

        ani = animation.FuncAnimation(fig, self.update_plot, frames=range(numframes),
                                      fargs=(scat,), interval=10, repeat=True)

        plt.show()

        return ani


if __name__ == '__main__':
    ani = FlockPlotter(Flock()).animate(1)
