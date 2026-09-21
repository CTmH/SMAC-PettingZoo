import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from s2clientprotocol import sc2api_pb2 as sc_pb
from smac_pettingzoo.env.render import StarCraft2Renderer
from smac_pettingzoo.smacv2_pettingzoo_v1 import ParallelEnv
from pysc2.lib import point


class TestRender(unittest.TestCase):
    def test_map_sizes_and_independent_rgb_frames(self):
        for width, height in ((32, 32), (88, 96), (120, 128)):
            info = sc_pb.ResponseGameInfo()
            info.start_raw.map_size.x = width
            info.start_raw.map_size.y = height
            info.start_raw.playable_area.p0.x = 12
            info.start_raw.playable_area.p0.y = 10
            info.start_raw.playable_area.p1.x = width - 12
            info.start_raw.playable_area.p1.y = height - 10
            env = SimpleNamespace(
                _controller=Mock(game_info=Mock(return_value=info)),
                window_size=(640, 480), _obs=sc_pb.ResponseObservation(),
                reward=0, _episode_steps=0, map_name="test",
                terrain_height=np.zeros((width, height)),
            )
            renderer = StarCraft2Renderer(env, "rgb_array")
            try:
                frame = renderer.render("rgb_array")
                self.assertEqual(frame.dtype, np.uint8)
                self.assertEqual(frame.shape, (480, round(width * 480 / height), 3))
                scale = 480 / height
                for x, y in ((0, height), (width, 0), (width / 2, height / 2), (12, 10)):
                    actual = renderer.screen_transform.fwd_pt(point.Point(x, y))
                    np.testing.assert_allclose(actual, (x * scale, (height - y) * scale))
                saved = frame.copy()
                renderer.display.fill((255, 0, 0))
                renderer.render("rgb_array")
                np.testing.assert_array_equal(frame, saved)
            finally:
                renderer.close()
                renderer.close()

    def test_public_interface(self):
        with self.assertRaises(ValueError):
            ParallelEnv("10gen_terran_2_vs_2", render_mode="invalid")
        env = ParallelEnv.__new__(ParallelEnv)
        env._env = Mock()
        env.states = None
        env.render_mode = "rgb_array"
        with self.assertRaises(RuntimeError):
            env.render()
        env.states = {}
        self.assertIs(env.render(), env._env.render.return_value)
        env._env.render.assert_called_once_with(mode="rgb_array")
        env.render_mode = None
        env._render_warning_issued = False
        with self.assertWarns(UserWarning):
            self.assertIsNone(env.render())
        self.assertIsNone(env.render())
