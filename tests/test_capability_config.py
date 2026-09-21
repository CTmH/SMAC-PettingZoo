import unittest
from copy import deepcopy

from smac_pettingzoo.smacv2_pettingzoo_v1 import ParallelEnv


class TestCapabilityConfig(unittest.TestCase):
    def test_distribution_override_rules(self):
        defaults = ParallelEnv._parse_capability_config("terran", 32, 32)
        original = deepcopy(defaults)
        merge = ParallelEnv._merge_capability_config
        for override in ({"p": 0.8}, {"dist_type": "surrounded_and_reflect", "p": 0.8}):
            result = merge(defaults, {"start_positions": override})
            self.assertEqual(result["start_positions"], defaults["start_positions"] | {"p": 0.8})
            self.assertEqual(result["team_gen"], defaults["team_gen"])

        overrides = {
            "start_positions": {"dist_type": "reflect_position", "map_x": 32, "map_y": 32},
            "team_gen": {"dist_type": "fixed_teams", "ally_team": {"marine": 32},
                         "enemy_team": {"marine": 32}, "observe": True},
        }
        result = merge(defaults, overrides)
        self.assertEqual(result, {"n_units": 32, "n_enemies": 32} | overrides)
        updated = merge(result, {"team_gen": {"ally_team": {"marauder": 32}}})
        self.assertEqual(updated["team_gen"]["ally_team"], {"marauder": 32})
        self.assertEqual(updated["team_gen"]["enemy_team"], {"marine": 32})
        result["team_gen"]["ally_team"]["marine"] = 0
        result["start_positions"]["env_key"] = "start_positions"
        self.assertEqual(overrides["team_gen"]["ally_team"], {"marine": 32})
        self.assertNotIn("env_key", overrides["start_positions"])
        result = merge(defaults, {"team_gen": {"weights": [1.0]}})
        self.assertEqual(result["team_gen"]["weights"], [1.0])
        result["team_gen"]["unit_types"].clear()
        self.assertEqual(defaults, original)
