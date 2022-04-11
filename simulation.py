import cupy as cp


class Flock(object):
    def __init__(self,
                 n_obj=1, x_scope=(0, 50), y_scope=(0, 50), abs_v=10, delta_t=0.1,
                 alpha_alignment=0.5, alpha_separation=1, alpha_cohesion=1, alpha_random=1, alpha_boundary_avoidance=20,
                 bound_threshold=13, r_vision=3, r_personal_space=1):
        self.n_obj = n_obj
        self.v_abs = abs_v
        self.delta_t = delta_t

        self.alpha_alignment = alpha_alignment
        self.alpha_separation = alpha_separation
        self.alpha_cohesion = alpha_cohesion
        self.alpha_random = alpha_random
        self.alpha_boundary_avoidance = alpha_boundary_avoidance
        self.bound_threshold = bound_threshold
        self.r_vision = r_vision
        self.r_personal_space = r_personal_space

        self.x_min, self.x_max = x_scope
        self.y_min, self.y_max = y_scope

        x = cp.random.rand(n_obj) * (self.x_max - self.x_min) + self.x_min
        y = cp.random.rand(n_obj) * (self.y_max - self.y_min) + self.y_min

        vx = (-1) ** cp.random.randint(1, 3, n_obj) * cp.random.rand(n_obj) * abs_v
        vy = (-1) ** cp.random.randint(1, 3, n_obj) * cp.sqrt(abs_v ** 2 - vx ** 2)

        self.state = cp.vstack((x, y, vx, vy)).T  # (n_obj, 4)

    def add_new_objs(self, new_obj_n):
        x = cp.random.rand(new_obj_n) * (self.x_max - self.x_min) + self.x_min
        y = cp.random.rand(new_obj_n) * (self.y_max - self.y_min) + self.y_min

        vx = (-1) ** cp.random.randint(1, 3, new_obj_n) * cp.random.rand(new_obj_n) * self.v_abs
        vy = (-1) ** cp.random.randint(1, 3, new_obj_n) * cp.sqrt(new_obj_n ** 2 - vx ** 2)

        new_obj_state = cp.vstack((x, y, vx, vy)).T  # (new_obj_n, 4)

        self.state = cp.vstack((self.state, new_obj_state))

    def delete_objs(self, del_obj_n):
        objs_to_stay_inds = cp.random.randint(0, self.n_obj, self.n_obj - del_obj_n)
        self.state = self.state[objs_to_stay_inds, :]

    def calc_distance_matrix(self):
        return cp.sqrt(((self.state[:, cp.newaxis, :2] - self.state[cp.newaxis, :, :2]) ** 2).sum(axis=2))

    def calc_neighbourhood_stats(self, distances, radius):
        neighbours = distances < radius
        cp.fill_diagonal(neighbours, False)

        neighbours_states_sum = (self.state[cp.newaxis, :, :] * neighbours[:, :, cp.newaxis]).sum(axis=1)

        neighbours_num = neighbours.sum(axis=1)
        has_neighbours = neighbours_num > 0

        neighbours_num = neighbours_num[cp.where(has_neighbours)[0], cp.newaxis]
        neighbours_states_sum = neighbours_states_sum[cp.where(has_neighbours)[0], :]

        return has_neighbours, neighbours_num, neighbours_states_sum

    def calc_alignment_v(self, has_visible_neighbours, visible_neighbours_num, visible_neighbours_states_sum):
        return self.state[cp.where(has_visible_neighbours)[0], 2:] * (1 - self.alpha_alignment) \
               + self.alpha_alignment / visible_neighbours_num * visible_neighbours_states_sum[:, 2:]

    def calc_cohesion_v(self, has_visible_neighbours, visible_neighbours_num, visible_neighbours_states_sum):
        return - self.alpha_cohesion * self.state[cp.where(has_visible_neighbours)[0], :2] \
               + (self.alpha_cohesion / visible_neighbours_num) * visible_neighbours_states_sum[:, :2]

    def calc_boundary_avoidance_v(self):
        distance_from_bounds = cp.abs(cp.vstack((self.state[:, 0] - self.x_min,
                                                 self.state[:, 0] - self.x_max,
                                                 self.state[:, 1] - self.y_min,
                                                 self.state[:, 1] - self.y_max)).T)

        closest_bound_inds = cp.argmin(distance_from_bounds, axis=1)
        min_distance_to_bound = cp.min(distance_from_bounds, axis=1)
        bound_changes = (cp.ones((2, self.n_obj)) / min_distance_to_bound).T * (
                (-1) ** (closest_bound_inds.reshape(-1, 1) % 2))
        bound_changes[:, 0] = bound_changes[:, 0] * (closest_bound_inds < 2)
        bound_changes[:, 1] = bound_changes[:, 1] * (closest_bound_inds >= 2)

        close_to_bound = min_distance_to_bound < self.bound_threshold

        return close_to_bound, bound_changes

    def calc_random_v(self):
        return (-1) ** cp.random.randint(1, 3, (self.n_obj, 2)) * cp.random.rand(self.n_obj, 2)

    def normalize_v(self):
        return self.v_abs * self.state[:, 2:] / cp.sqrt(self.state[:, 2] ** 2 + self.state[:, 3] ** 2).reshape((-1, 1))

    def update_v(self, distances):
        has_visible_neighbours, visible_neighbours_num, visible_neighbours_states_sum = \
            self.calc_neighbourhood_stats(distances, self.r_vision)

        self.state[cp.where(has_visible_neighbours)[0], 2:] = \
            self.calc_alignment_v(has_visible_neighbours, visible_neighbours_num, visible_neighbours_states_sum) \
            + self.calc_cohesion_v(has_visible_neighbours, visible_neighbours_num, visible_neighbours_states_sum)

        has_close_neighbours, close_neighbours_num, close_neighbours_states_sum = \
            self.calc_neighbourhood_stats(distances, self.r_personal_space)

        self.state[cp.where(has_close_neighbours)[0], 2:] = \
            self.state[cp.where(has_close_neighbours)[0], 2:] \
            + self.state[cp.where(has_close_neighbours)[0], :2] * self.alpha_separation * close_neighbours_num \
            - self.alpha_separation * close_neighbours_states_sum[:, :2]

        close_to_bound, bound_changes = self.calc_boundary_avoidance_v()
        self.state[cp.where(close_to_bound)[0], 2:] = \
            self.state[cp.where(close_to_bound)[0], 2:] \
            + self.alpha_boundary_avoidance * bound_changes[cp.where(close_to_bound)[0]]

        self.state[:, 2:] = self.state[:, 2:] + \
                            self.alpha_random * self.calc_random_v()

        self.state[:, 2:] = self.normalize_v()

    def enforce_periodic_boundaries(self):
        x_left_mask = self.state[:, 0] < self.x_min
        self.state[cp.where(x_left_mask)[0], 0] = self.x_max - (self.x_min - self.state[cp.where(x_left_mask)[0], 0])
        x_right_mask = self.state[:, 0] > self.x_max
        self.state[cp.where(x_right_mask)[0], 0] = self.x_min + (self.x_max - self.state[cp.where(x_right_mask)[0], 0])

        y_down_mask = self.state[:, 1] < self.y_min
        self.state[cp.where(y_down_mask)[0], 1] = self.y_max - (self.y_min - self.state[cp.where(y_down_mask)[0], 1])
        y_up_mask = self.state[:, 1] > self.y_max
        self.state[cp.where(y_up_mask)[0], 1] = self.y_min + (self.y_max - self.state[cp.where(y_up_mask)[0], 1])

    def update_r(self):
        self.state[:, :2] = self.state[:, :2] + self.delta_t * self.state[:, 2:]
        self.enforce_periodic_boundaries()

    def iterate(self, n_iter):
        log = [self.state.copy()]
        for _ in range(n_iter):
            distances = self.calc_distance_matrix()

            self.update_v(distances)
            self.update_r()
            log.append(self.state.copy())

        return cp.array(log)
