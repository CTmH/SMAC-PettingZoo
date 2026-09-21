"""Standalone, map-size-aware SMACv2 renderer.

Adapted from CTmH/smacv2, commit 2e869a6bef4144516daea993e4b0bed877a8c618,
smacv2/env/starcraft2/render.py (originally oxwhirl/smacv2).
"""
# MIT License
#
# Copyright (c) 2019 whirl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import collections
import math
import time
import pygame
from pysc2.lib import colors, features, point, transform
from pysc2.lib.renderer_human import _Surface


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


class StarCraft2Renderer:
    def __init__(self, env, mode):
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"Unsupported render mode: {mode!r}")
        size = env.window_size
        if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError("window_size must contain two positive integers")
        self.env = env
        self.static_data = env._controller.data()
        self._map_size = point.Point.build(env._controller.game_info().start_raw.map_size)
        self._game_times = collections.deque(maxlen=100)
        self._render_times = collections.deque(maxlen=100)
        self._last_time = time.monotonic()
        self._last_game_loop = env._obs.observation.game_loop
        self._name_lengths = {}
        self.upgrade_colors = (colors.white, colors.yellow, colors.Color(255, 128, 0), colors.red)
        self.mode = mode
        width, height = self._map_size
        scale = min(size[0] / width, size[1] / height)
        pixels = (max(1, round(width * scale)), max(1, round(height * scale)))
        self.display = pygame.Surface(pixels)
        self._window_closed = False
        if mode == "human":
            try:
                pygame.display.init()
                self.display = pygame.display.set_mode(pixels)
                pygame.display.set_caption("Starcraft Viewer")
            except pygame.error as exc:
                pygame.display.quit()
                raise RuntimeError("human rendering requires a working graphical display; use rgb_array on headless servers.") from exc
        # Fit the entire map, including nonzero playable-area offsets. Use a
        # single scale for both axes; pixel rounding may leave a subpixel margin.
        self._world_tl_to_screen = transform.Linear(scale=scale)
        self._world_to_world_tl = transform.Linear(point.Point(1, -1), point.Point(0, height))
        self.screen_transform = transform.Chain(self._world_to_world_tl, self._world_tl_to_screen)
        self._surf = _Surface(self.display, None, point.Rect(point.origin, point.Point(*pixels)),
                              self.screen_transform, None, self.draw_screen)
        pygame.font.init()
        self._font_small = pygame.font.Font(None, max(8, round(scale * 0.5)))
        self._font_large = pygame.font.Font(None, max(12, round(scale)))

    def render(self, mode):
        if mode != self.mode:
            raise ValueError("Render mode must remain unchanged for this renderer.")
        if mode == "human":
            if self._window_closed:
                return None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
        self.obs = self.env._obs
        self.score = self.env.reward
        self.step = self.env._episode_steps
        now = time.monotonic()
        game_loop = self.obs.observation.game_loop
        if game_loop < self._last_game_loop:
            self._game_times.clear()
        self._game_times.append((now - self._last_time, max(0, game_loop - self._last_game_loop)))
        self._surf.draw(self._surf)
        self._render_times.append(time.monotonic() - now)
        self._last_time = now
        self._last_game_loop = game_loop
        if mode == "human":
            pygame.display.flip()
            return None
        return pygame.surfarray.array3d(self.display).transpose(1, 0, 2).copy()

    def close(self):
        # Do not globally quit pygame/font: other off-screen renderers may exist.
        if self.mode == "human" and not self._window_closed:
            pygame.display.quit()
        self._window_closed = True

    def _get_units(self):
        for u in sorted(
            self.obs.observation.raw_data.units,
            key=lambda u: (u.pos.z, u.owner != 16, -u.radius, u.tag),
        ):
            yield u, point.Point.build(u.pos)

    def get_unit_name(self, surf, name, radius):
        """Get a length limited unit name for drawing units."""
        key = (name, radius)
        if key not in self._name_lengths:
            max_len = surf.world_to_surf.fwd_dist(radius * 1.6)
            for i in range(len(name)):
                if self._font_small.size(name[: i + 1])[0] > max_len:
                    self._name_lengths[key] = name[:i]
                    break
            else:
                self._name_lengths[key] = name
        return self._name_lengths[key]

    def draw_base_map(self, surf):
        """Draw the base map."""
        hmap_feature = features.SCREEN_FEATURES.height_map
        hmap = self.env.terrain_height * 255
        hmap = hmap.astype(np.uint8)
        if (
            self.env.map_name == "corridor"
            or self.env.map_name == "so_many_baneling"
            or self.env.map_name == "2s_vs_1sc"
        ):
            hmap = np.flip(hmap)
        else:
            hmap = np.rot90(hmap, axes=(1, 0))
        if not hmap.any():
            hmap = hmap + 100  # pylint: disable=g-no-augmented-assignment
        hmap_color = hmap_feature.color(hmap)
        out = hmap_color * 0.6

        surf.blit_np_array(out)

    def draw_units(self, surf):
        """Draw the units."""
        unit_dict = None  # Cache the units {tag: unit_proto} for orders.
        tau = 2 * math.pi
        for u, p in self._get_units():
            fraction_damage = clamp(
                (u.health_max - u.health) / (u.health_max or 1), 0, 1
            )
            surf.draw_circle(
                colors.PLAYER_ABSOLUTE_PALETTE[u.owner], p, u.radius
            )

            if fraction_damage > 0:
                surf.draw_circle(
                    colors.PLAYER_ABSOLUTE_PALETTE[u.owner] // 2,
                    p,
                    u.radius * fraction_damage,
                )
            surf.draw_circle(colors.black, p, u.radius, thickness=1)

            if self.static_data.unit_stats[u.unit_type].movement_speed > 0:
                surf.draw_arc(
                    colors.white,
                    p,
                    u.radius,
                    u.facing - 0.1,
                    u.facing + 0.1,
                    thickness=1,
                )

            def draw_arc_ratio(
                color, world_loc, radius, start, end, thickness=1
            ):
                surf.draw_arc(
                    color, world_loc, radius, start * tau, end * tau, thickness
                )

            if u.shield and u.shield_max:
                draw_arc_ratio(
                    colors.blue, p, u.radius - 0.05, 0, u.shield / u.shield_max
                )

            if u.energy and u.energy_max:
                draw_arc_ratio(
                    colors.purple * 0.9,
                    p,
                    u.radius - 0.1,
                    0,
                    u.energy / u.energy_max,
                )
            elif u.orders and 0 < u.orders[0].progress < 1:
                draw_arc_ratio(
                    colors.cyan, p, u.radius - 0.15, 0, u.orders[0].progress
                )
            if u.buff_duration_remain and u.buff_duration_max:
                draw_arc_ratio(
                    colors.white,
                    p,
                    u.radius - 0.2,
                    0,
                    u.buff_duration_remain / u.buff_duration_max,
                )
            if u.attack_upgrade_level:
                draw_arc_ratio(
                    self.upgrade_colors[u.attack_upgrade_level],
                    p,
                    u.radius - 0.25,
                    0.18,
                    0.22,
                    thickness=3,
                )
            if u.armor_upgrade_level:
                draw_arc_ratio(
                    self.upgrade_colors[u.armor_upgrade_level],
                    p,
                    u.radius - 0.25,
                    0.23,
                    0.27,
                    thickness=3,
                )
            if u.shield_upgrade_level:
                draw_arc_ratio(
                    self.upgrade_colors[u.shield_upgrade_level],
                    p,
                    u.radius - 0.25,
                    0.28,
                    0.32,
                    thickness=3,
                )

            def write_small(loc, s):
                surf.write_world(self._font_small, colors.white, loc, str(s))

            name = self.get_unit_name(
                surf,
                self.static_data.units.get(u.unit_type, "<none>"),
                u.radius,
            )

            if name:
                write_small(p, name)

            start_point = p
            for o in u.orders:
                target_point = None
                if o.HasField("target_unit_tag"):
                    if unit_dict is None:
                        unit_dict = {
                            t.tag: t
                            for t in self.obs.observation.raw_data.units
                        }
                    target_unit = unit_dict.get(o.target_unit_tag)
                    if target_unit:
                        target_point = point.Point.build(target_unit.pos)
                if target_point:
                    surf.draw_line(colors.cyan, start_point, target_point)
                    start_point = target_point
                else:
                    break

    def draw_overlay(self, surf):
        """Draw the overlay describing resources."""
        obs = self.obs.observation
        times, steps = zip(*self._game_times)
        sec = obs.game_loop // 22.4
        surf.write_screen(
            self._font_large,
            colors.green,
            (-0.2, 0.2),
            "Score: %s, Step: %s, %.1f/s, Time: %d:%02d"
            % (
                self.score,
                self.step,
                sum(steps) / (sum(times) or 1),
                sec // 60,
                sec % 60,
            ),
            align="right",
        )
        surf.write_screen(
            self._font_large,
            colors.green * 0.8,
            (-0.2, 1.2),
            "APM: %d, EPM: %d, FPS: O:%.1f, R:%.1f"
            % (
                obs.score.score_details.current_apm,
                obs.score.score_details.current_effective_apm,
                len(times) / (sum(times) or 1),
                len(self._render_times) / (sum(self._render_times) or 1),
            ),
            align="right",
        )

    def draw_screen(self, surf):
        """Draw the screen area."""
        self.draw_base_map(surf)
        self.draw_units(surf)
        self.draw_overlay(surf)
