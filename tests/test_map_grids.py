"""Non-square SC2 grids must still support lookup by [x, y]."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from s2clientprotocol import sc2api_pb2 as sc_pb
from smac_pettingzoo.env.smacv2 import SMACv2EnvCore


class TestMapGrids(unittest.TestCase):
    def test_rectangular_grid_and_bit_padding(self):
        for bits in (1, 8):
            with self.subTest(bits=bits):
                env = SMACv2EnvCore.__new__(SMACv2EnvCore)
                env.map_name, env.map_file, env.game_version = "10gen_terran", None, None
                env._seed, env.window_size = 0, (640, 480)
                env._bot_race = env._agent_race = "T"
                env.difficulty = "7"
                info = sc_pb.ResponseGameInfo()
                raw = info.start_raw
                raw.map_size.x, raw.map_size.y = 5, 3
                path = np.arange(15, dtype=np.uint8).reshape(3, 5) % 2
                raw.pathing_grid.bits_per_pixel = bits
                raw.pathing_grid.data = np.packbits(path).tobytes() if bits == 1 else path.tobytes()
                height = np.arange(15, dtype=np.uint8).reshape(3, 5)
                raw.terrain_height.data = height.tobytes()
                run_config = Mock()
                run_config.map_data.return_value = b"map"
                run_config.start.return_value.controller.game_info.return_value = info
                with patch("smac_pettingzoo.env.smacv2.run_configs.get", return_value=run_config), \
                     patch("smac_pettingzoo.env.smacv2.maps.get", return_value=SimpleNamespace(path="map.SC2Map")):
                    env._launch()
                expected = path.T.astype(bool) if bits == 1 else ~path.T[:, ::-1].astype(bool)
                np.testing.assert_array_equal(env.pathing_grid, expected)
                np.testing.assert_array_equal(env.terrain_height, height.T[:, ::-1] / 255)
