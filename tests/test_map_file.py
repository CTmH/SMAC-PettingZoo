"""Map-file overrides do not require launching SC2 in unit tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from smac_pettingzoo.env.smacv2 import SMACv2EnvCore


class TestMapFile(unittest.TestCase):
    def test_invalid_paths(self):
        for path in ("", " ", "Flat64", 64):
            with self.subTest(path=path), self.assertRaises(ValueError):
                SMACv2EnvCore("10gen_terran", {}, map_file=path)

    def test_default_and_override_are_forwarded_before_launch(self):
        for override in (None, "Melee/Flat64.SC2Map", "/tmp/Flat64.SC2Map"):
            with self.subTest(override=override):
                env = SMACv2EnvCore.__new__(SMACv2EnvCore)
                env.map_name = "10gen_terran"
                env.map_file = override
                env.game_version = None
                env._seed = 0
                env.window_size = (640, 480)
                env._bot_race = env._agent_race = "T"
                env.difficulty = "7"
                controller = Mock()
                # Stop after create_game: terrain-grid parsing is unrelated.
                controller.join_game.side_effect = RuntimeError("stop after create")
                run_config = Mock()
                run_config.map_data.return_value = b"map bytes"
                run_config.start.return_value.controller = controller
                with patch("smac_pettingzoo.env.smacv2.run_configs.get", return_value=run_config), \
                     patch("smac_pettingzoo.env.smacv2.maps.get", return_value=SimpleNamespace(path="default.SC2Map")):
                    with self.assertRaisesRegex(RuntimeError, "stop after create"):
                        env._launch()
                expected = override or "default.SC2Map"
                run_config.map_data.assert_called_once_with(expected)
                request = controller.create_game.call_args.args[0]
                self.assertEqual(request.local_map.map_path, expected)
                self.assertEqual(request.local_map.map_data, b"map bytes")
                self.assertEqual(env.map_name, "10gen_terran")

    def test_missing_map_does_not_start_sc2(self):
        env = SMACv2EnvCore.__new__(SMACv2EnvCore)
        env.map_name, env.map_file, env.game_version = "10gen_terran", "missing.SC2Map", None
        run_config = Mock()
        run_config.map_data.side_effect = ValueError("map missing")
        with patch("smac_pettingzoo.env.smacv2.run_configs.get", return_value=run_config):
            with self.assertRaisesRegex(ValueError, "map missing"):
                env._launch()
        run_config.start.assert_not_called()
