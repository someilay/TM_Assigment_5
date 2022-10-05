import pickle
from math import sqrt, cos, sin, inf, asin
from manim import *
from scipy.integrate import solve_ivp
from typing import Callable, Optional
from helpers import get_axes_config, get_mut_dot, get_dot_title, create_graph_law, c2p, get_vector, get_vector_title, \
    get_timer, part_apply, sub_slice, bin_search


class Task1(Scene):
    M = 13.6 * 1e-3
    L = 1500
    K = 1.3 * 1e-5
    G = 10
    V0 = 870
    X0 = Y0 = 0
    K_OVER_M = K / M

    ALPHA_1_W_A_R = asin(L * G / V0 ** 2) / 2
    ALPHA_2_W_A_R = PI / 2 - ALPHA_1_W_A_R
    TOTAL_TIME_1_W_A_R = 2 * V0 * sin(ALPHA_1_W_A_R) / G
    TOTAL_TIME_2_W_A_R = 2 * V0 * sin(ALPHA_2_W_A_R) / G
    INIT_TIME = 0

    N_SUP_POINTS = 10 ** 6
    SOLS_WITH_A_RES = []

    # Vector configs
    VECTORS_KWARGS = {
        'stroke_width': 3,
        'tip_length': 0.25,
        'buff': 0,
        'max_tip_length_to_length_ratio': 0.2,
        'max_stroke_width_to_length_ratio': 10,
    }

    GRAPH_CONFIGS = [
        {
            'x_length': 6,
            'y_length': 6,
        }, {
            'x_length': 3.5,
            'y_length': 3,
            'x_decimal_place': 1,
            'y_decimal_place': 1,
        }, {
            'x_length': 3.5,
            'y_length': 3,
            'x_decimal_place': 1,
            'y_decimal_place': 1,
        },
    ]

    @staticmethod
    def y_from_x_w_a_r(x: float, alpha: float) -> float:
        source = Task1
        t = (x - source.X0) / (source.V0 * cos(alpha))
        return source.coord_w_a_r(t, alpha)[1]

    @staticmethod
    def coord_w_a_r(t: float, alpha: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                source.X0 + source.V0 * cos(alpha) * t,
                source.Y0 + source.V0 * sin(alpha) * t - source.G * (t ** 2) / 2,
                0
            )
        )

    @staticmethod
    def vel_w_a_r(t: float, alpha: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                source.V0 * cos(alpha),
                source.V0 * sin(alpha) - source.G * t,
                0
            )
        )

    @staticmethod
    def acc_w_a_r(t: float, alpha: float) -> np.ndarray:
        source = Task1
        return np.array(
            (
                0,
                -source.G,
                0
            )
        )

    @staticmethod
    def find_map(from_: np.ndarray, to_: np.ndarray,
                 val: Optional[float] = None, idx: Optional[int] = None) -> tuple[int, float]:
        if idx is not None:
            return idx, to_[idx]
        if val is None:
            raise ValueError('At least val should not be None, when idx is None')

        idx = bin_search(from_, val)
        cur_v, prev_v = val, val
        cur_m, pred_m = 0, 0

        if idx < len(from_):
            cur_m, cur_v = to_[idx], from_[idx]
        if idx > 0:
            pred_m, prev_v = to_[idx - 1], from_[idx - 1]

        if idx == 0:
            return idx, cur_m
        if idx == len(from_):
            return idx, pred_m

        return idx, (cur_m - pred_m) * (val - prev_v) / (cur_v - prev_v) + pred_m

    @staticmethod
    def param_a_r(t: float, degree: int, sol: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
        if degree not in range(2):
            raise ValueError(f'Unsupported degree = {degree}. Supported degrees are in the range [0,1]!')

        source = Task1
        _, timestamps, states = sol
        idx, f_map = source.find_map(timestamps, states[2 * degree], val=t)
        _, s_map = source.find_map(timestamps, states[2 * degree + 1], idx=idx)
        return np.array((f_map, s_map, 0))

    @staticmethod
    def y_from_x_a_r(x: float, sol: tuple[float, np.ndarray, np.ndarray]) -> float:
        source = Task1
        _, _, states = sol
        return source.find_map(states[0], states[1], val=x)[1]

    @staticmethod
    def coord_a_r(t: float, sol: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
        source = Task1
        return source.param_a_r(t, 0, sol)

    @staticmethod
    def vel_a_r(t: float, sol: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
        source = Task1
        return source.param_a_r(t, 1, sol)

    @staticmethod
    def acc_a_r(t: float, sol: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
        source = Task1
        vx, vy, _ = source.vel_a_r(t, sol)
        return np.array(
            (
                -source.K_OVER_M * vx * sqrt(vx ** 2 + vy ** 2),
                -source.K_OVER_M * vy * sqrt(vx ** 2 + vy ** 2) - source.G,
                0,
            )
        )

    @staticmethod
    def force_r(t: float, sol: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
        source = Task1
        vx, vy, _ = source.vel_a_r(t, sol)
        return np.array(
            (
                -source.K_OVER_M * vx * sqrt(vx ** 2 + vy ** 2) * source.M,
                -source.K_OVER_M * vy * sqrt(vx ** 2 + vy ** 2) * source.M,
                0,
            )
        )

    @staticmethod
    def state_model(t: float, y: np.ndarray, k_over_m: float, g: float) -> np.ndarray:
        _, _, vx, vy = y
        v = sqrt(vx ** 2 + vy ** 2)
        return np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, -k_over_m * v, 0],
                [0, 0, 0, -k_over_m * v]
            ]
        ).dot(y) + np.array(
            [0, 0, 0, -g]
        )

    def simulate_with_air_resistance(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        def stop_condition(t, y, *args) -> float:
            if t > 0 > y[1]:
                return -1
            return 1

        stop_condition.terminal = True

        y_init = np.array([self.X0, self.Y0, self.V0 * cos(alpha), self.V0 * sin(alpha)])
        t_span = (0, 100)
        t_eval = np.linspace(*t_span, self.N_SUP_POINTS)

        sol = solve_ivp(
            self.state_model, t_span, y_init, args=(self.K_OVER_M, self.G), t_eval=t_eval, events=stop_condition
        )
        # noinspection PyUnresolvedReferences
        return sol.t, sol.y

    @staticmethod
    def get_saddle_intervals(values: list[float], coords: np.ndarray) -> list[tuple[float, float]]:
        res = []
        start_coord = coords[0]
        for prev, cur, nxt, prev_coord, nxt_coord in zip(values, values[1:], values[2:], coords, coords[2:]):
            quadruple = [start_coord, prev_coord, nxt_coord]
            if prev >= cur <= nxt:
                for start, end in zip(quadruple, quadruple[1:]):
                    if start < end:
                        res.append((start, end))
                start_coord = nxt_coord
        if start_coord > coords[0]:
            res.append((start_coord, coords[-1]))
        return res

    @staticmethod
    def sew_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        idx = 0
        while idx < len(intervals):
            if idx != len(intervals) - 1 and intervals[idx][1] == intervals[idx + 1][0]:
                intervals[idx + 1] = (intervals[idx][0], intervals[idx + 1][1])
                intervals.pop(idx)
                continue
            idx += 1
        return intervals

    def check_saved(self, *opt_params) -> list[tuple[float, np.ndarray, np.ndarray]]:
        try:
            with open('cashed.pickle', 'rb') as f:
                cashed = pickle.load(f)
            return cashed.get(
                (self.M, self.L, self.K, self.G, self.V0, self.X0, self.Y0, self.N_SUP_POINTS, *opt_params),
                []
            )
        except FileNotFoundError:
            return []

    def save_res(self, res: list[tuple[float, np.ndarray, np.ndarray]], *opt_params):
        try:
            with open('cashed.pickle', 'rb') as f:
                cashed = pickle.load(f)
        except FileNotFoundError:
            cashed = dict()

        cashed[(self.M, self.L, self.K, self.G, self.V0, self.X0, self.Y0, self.N_SUP_POINTS, *opt_params)] = res
        with open('cashed.pickle', 'wb') as f:
            pickle.dump(cashed, f)

    def solve_with_air_resistance(self) -> list[tuple[float, np.ndarray, np.ndarray]]:
        start_g = 0
        end_g = PI / 2
        intervals = [(start_g, end_g)]
        n_points = 10
        n_iterations = 8

        res = self.check_saved(start_g, end_g, n_points, n_iterations)
        if res:
            return res

        for _ in range(n_iterations):
            new_intervals = []
            for start, end in intervals:
                alphas = np.linspace(start, end, n_points)
                values = []
                for alpha in alphas:
                    _, ys = self.simulate_with_air_resistance(alpha)
                    values.append(abs(self.L - ys[0][-1]))
                new_intervals.extend(self.get_saddle_intervals(values, alphas))
            intervals = new_intervals

        intervals = self.sew_intervals(intervals)
        for start, end in intervals:
            alphas = np.linspace(start, end, n_points)
            min_alpha = alphas[0]
            min_value = inf
            min_ts = None
            min_ys = None
            for alpha in alphas:
                ts, ys = self.simulate_with_air_resistance(alpha)
                if abs(ys[0][-1] - self.L) < min_value:
                    min_value = abs(ys[0][-1] - self.L)
                    min_alpha = alpha
                    min_ts = ts
                    min_ys = ys
            res.append((min_alpha, min_ts, min_ys))

        self.save_res(res, start_g, end_g, n_points, n_iterations)
        return res

    def show_text(self, text: str, duration: float, font_size: float):
        self.clear_scene()
        self.play(Write(Text(text, font_size=font_size)))
        self.wait(duration)

    def clear_scene(self):
        self.remove(*self.mobjects)

    @staticmethod
    def get_params(lin_space: np.ndarray) -> np.ndarray:
        return np.array([lin_space[0], lin_space[-1], lin_space[1] - lin_space[0]])

    @staticmethod
    def get_step_size(min_v: float, max_v: float, n_steps: int) -> float:
        d_step = (max_v - min_v) / n_steps
        if d_step >= 1:
            step = round(d_step)
            return round(step, 2 - len(str(step)))
        else:
            return round(d_step, 1)

    def create_axes(self, funcs: list[Callable[[float], float]], x_range: np.ndarray, font_size: int, x_length: float,
                    y_length: float, x_decimal_place: int = 0, y_decimal_place: int = 0) -> Axes:
        n_x_labels = int(10 * x_length / 6)
        n_y_labels = int(10 * y_length / 6)

        y_ranges = np.array([[func(x) for x in x_range] for func in funcs])
        x_step = self.get_step_size(x_range[0], x_range[-1], n_x_labels)
        y_step = self.get_step_size(np.amin(y_ranges), np.amax(y_ranges), n_y_labels)

        axes = Axes(
            **get_axes_config(
                x_range=[x_range[0] - x_step / 2, x_range[-1] + x_step / 2, x_step],
                y_range=[np.amin(y_ranges) - y_step / 2, np.amax(y_ranges) + y_step / 2, y_step],
                x_step=x_step,
                y_step=y_step,
                font_size=font_size,
                x_length=x_length,
                y_length=y_length,
                x_decimal_place=x_decimal_place,
                y_decimal_place=y_decimal_place,
            )
        )
        return axes

    def create_graph(self, axes: Axes, funcs: list[Callable[[float], float]],
                     x_range: np.ndarray, colors: list[str], stroke_width: float, ) -> list[ParametricFunction]:
        return [
            axes.plot(func, x_range=self.get_params(x_range), stroke_width=stroke_width, color=c)
            for func, c in zip(funcs, colors)
        ]

    @staticmethod
    def create_axis_labels(axes: Axes, x_text: str, y_text: str, font_size: int) -> list[MathTex]:
        xa_label, ya_label = axes.get_axis_labels(x_label=MathTex(x_text, font_size=font_size),
                                                  y_label=MathTex(y_text, font_size=font_size))
        return [xa_label, ya_label]

    @staticmethod
    def create_graph_dots(vt: ValueTracker, axes: Axes, decompose: bool,
                          law: Callable[[float], np.ndarray], c: str, radius: float) -> list[Dot]:
        if decompose:
            return [
                get_mut_dot(vt, axes, c, radius, create_graph_law(law, idx))
                for idx in range(2)
            ]
        else:
            return [get_mut_dot(vt, axes, c, radius, law)]

    @staticmethod
    def create_dot_tiles(vt: ValueTracker, axes: Axes, titles: list[str], font_size: int,
                         law: Callable[[float], np.ndarray], colors: list[str]) -> list[MathTex]:
        return [
            get_dot_title(
                vt, axes, title, font_size,
                create_graph_law(law, idx),
                lambda _: UR / 5, c
            )
            for idx, (title, c) in enumerate(zip(titles, colors))
        ]

    @staticmethod
    def handle_kwargs(**kwargs) -> list[tuple]:
        endings = ['force_law', 'force_col', 'force_name', 'force_scale']

        def is_encoded_suffix(obj: str) -> tuple[bool, str, str]:
            for ending in endings:
                if obj.endswith(ending):
                    return True, obj[:-len(ending)], ending
            return False, obj, obj

        info = dict()
        for key, value in kwargs.items():
            is_encoded, id_, class_ = is_encoded_suffix(key)
            if is_encoded:
                if id_ in info:
                    info[id_][class_] = value
                else:
                    info[id_] = {class_: value}

        for key, value in list(info.items()):
            for subkey in value.keys():
                if subkey not in endings or len(value.keys()) != len(endings):
                    info.pop(key, None)
                    break

        res = []
        for key, value in info.items():
            res.append(
                tuple(
                    [value[ending] for ending in endings]
                )
            )

        return res

    def mark_max_height(self, axes: Axes, y_from_x: Callable[[float], float],
                        x_range: np.ndarray, font_size: int) -> tuple[DashedVMobject, MathTex]:
        y_range = np.array([y_from_x(x) for x in x_range])
        max_y = np.max(y_range)
        max_x = x_range[np.argmax(y_range)]

        d_arrow = DoubleArrow(
            start=c2p(axes, np.array((max_x, 0, 0))),
            end=c2p(axes, np.array((max_x, max_y, 0))),
            **self.VECTORS_KWARGS
        )
        max_h = MathTex(f'{max_y:.2f},m', font_size=font_size)
        max_h.next_to(d_arrow, RIGHT * 0.3)
        return DashedVMobject(d_arrow), max_h

    def scene(self, y_from_x: Callable[[float], float], coord_from_t: Callable[[float], np.ndarray],
              v_from_t: Callable[[float], np.ndarray],
              a_from_t: Callable[[float], np.ndarray], x_ranges: list[np.ndarray],
              init_time: float, end_time: float, p_speed: float,
              alpha: float, p_font_size: float, p_dur: float, is_air_res: bool,
              vel_scale: float, g_scale: float, **kwargs):
        air_res_text = 'with' if is_air_res else 'without'
        self.show_text(
            f'Simulation {air_res_text} air resistance\n'
            f'Initial angle = {alpha:.5f} rads., playback speed = {p_speed:.2f}',
            p_dur, p_font_size
        )
        self.clear_scene()

        vt = ValueTracker(init_time)

        all_laws = [[y_from_x]] + [[sub_slice(law, d) for d in range(2)] for law in [v_from_t, a_from_t]]

        positions = [3.5 * LEFT, 3.5 * RIGHT + 1.9 * UP, 3.5 * RIGHT + 1.9 * DOWN]
        timer_pos = 3.5 * UP

        ax_labels = [['y,m', 'x,m'], ['t,s', 'V,\\frac{m}{s}'], ['t,s', 'a,\\frac{m}{s^2}']]
        dot_titles = [['V_x', 'V_y'], ['a_x', 'a_y']]
        vec_titles = ['\\Vec{V}', '\\Vec{G}']
        timer_num_dec_places = 2
        stroke_width = DEFAULT_STROKE_WIDTH / 2
        dot_radius = DEFAULT_DOT_RADIUS / 2

        colors = [[GREEN], [BLUE, BLUE_B], [RED, RED_B]]
        dot_color = WHITE

        font_size = 16
        timer_font_size = 28
        axes_labels_font_size = 20
        dot_title_font = 20
        vec_font_size = 20

        all_axes = []
        all_graphs = []
        all_axes_labels = []
        all_dots = []
        all_dots_titles = []
        all_vectors = []
        all_vectors_titles = []

        for sub_laws, position, x_range, graph_config in zip(all_laws, positions, x_ranges, self.GRAPH_CONFIGS):
            all_axes.append(
                self.create_axes(
                    funcs=sub_laws, x_range=x_range, font_size=font_size, **graph_config
                ).move_to(position)
            )
        for axes, sub_laws, x_range, sub_colors in zip(all_axes, all_laws, x_ranges, colors):
            all_graphs.extend(
                self.create_graph(axes, sub_laws, x_range, sub_colors, stroke_width)
            )
        for axes, (x_text, y_text) in zip(all_axes, ax_labels):
            all_axes_labels.extend(
                self.create_axis_labels(axes, x_text, y_text, axes_labels_font_size)
            )
        for idx, (axes, law) in enumerate(zip(all_axes, [coord_from_t, v_from_t, a_from_t])):
            all_dots.extend(
                self.create_graph_dots(vt, axes, idx != 0, law, dot_color, dot_radius)
            )
        for axes, law, sub_titles, sub_colors in zip(all_axes[1:], [v_from_t, a_from_t], dot_titles, colors[1:]):
            all_dots_titles.extend(
                self.create_dot_tiles(vt, axes, sub_titles, dot_title_font, law, sub_colors)
            )

        all_vectors.append(get_vector(vt, all_axes[0], BLUE, coord_from_t, v_from_t, vel_scale, self.VECTORS_KWARGS))
        all_vectors.append(
            get_vector(
                vt, all_axes[0], GREEN, coord_from_t,
                lambda ct: np.array((0, -self.M * self.G, 0)), g_scale, self.VECTORS_KWARGS
            )
        )
        add_vectors_info = self.handle_kwargs(**kwargs)
        for force_law, force_col, force_name, force_scale in add_vectors_info:
            vec_titles.append(force_name)
            all_vectors.append(
                get_vector(
                    vt, all_axes[0], force_col, coord_from_t, force_law, force_scale, self.VECTORS_KWARGS
                )
            )
        for vector, vec_title in zip(all_vectors, vec_titles):
            all_vectors_titles.append(
                get_vector_title(vt, vector, vec_title, vec_font_size, lambda vec: normalize(vec) * 0.2)
            )

        timer = get_timer(vt, timer_pos, timer_font_size, num_decimal_places=timer_num_dec_places)
        max_height = self.mark_max_height(all_axes[0], y_from_x, x_ranges[0], vec_font_size)

        self.add(timer)
        self.add(*all_axes)
        self.add(*all_graphs)
        self.add(*max_height)
        self.add(*all_axes_labels)
        self.add(*all_vectors)
        self.add(*all_vectors_titles)
        self.add(*all_dots)
        self.add(*all_dots_titles)
        self.play(
            vt.animate.set_value(end_time),
            run_time=(end_time - init_time) / p_speed,
            rate_func=linear
        )

    def construct(self):
        sol_with_air_res = self.solve_with_air_resistance()
        preamble_dur = 0.7
        preamble_font_size = 0.7 * DEFAULT_FONT_SIZE
        n_points = 10 ** 3
        alphas = [self.ALPHA_1_W_A_R, self.ALPHA_2_W_A_R] + [alpha for alpha, _, _ in sol_with_air_res]
        t_times = [self.TOTAL_TIME_1_W_A_R, self.TOTAL_TIME_2_W_A_R] + [ts[-1] for _, ts, _ in sol_with_air_res]
        scales_1 = [(0.25, 5), (10, 50000)]
        scales_2 = [(0.4, 25, 20), (3, 2000, 500)]
        p_speeds = [0.25, 15, 0.33, 4]

        for alpha, total_time, (v_scale, g_scale), p_speed in zip(alphas, t_times, scales_1, p_speeds):
            self.scene(
                part_apply(self.y_from_x_w_a_r, alpha),
                part_apply(self.coord_w_a_r, alpha),
                part_apply(self.vel_w_a_r, alpha),
                part_apply(self.acc_w_a_r, alpha),
                [np.linspace(self.X0, self.L, n_points)] + [np.linspace(self.INIT_TIME, total_time, n_points)] * 2,
                self.INIT_TIME, total_time, p_speed,
                alpha, preamble_font_size, preamble_dur, False,
                v_scale, g_scale
            )

        for alpha, total_time, (v_scale, g_scale, r_scale), p_speed, sol in \
                zip(alphas[2:], t_times[2:], scales_2, p_speeds[2:], sol_with_air_res):
            self.scene(
                part_apply(self.y_from_x_a_r, sol),
                part_apply(self.coord_a_r, sol),
                part_apply(self.vel_a_r, sol),
                part_apply(self.acc_a_r, sol),
                [np.linspace(self.X0, self.L, n_points)] + [np.linspace(self.INIT_TIME, total_time, n_points)] * 2,
                self.INIT_TIME, total_time, p_speed,
                alpha, preamble_font_size, preamble_dur, True,
                v_scale, g_scale,
                res_force_law=part_apply(self.force_r, sol),
                res_force_col=RED, res_force_name='\\Vec{F}_r', res_force_scale=r_scale
            )
