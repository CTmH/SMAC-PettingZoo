import unittest
import numpy as np

from smac_pettingzoo.env.smacv2_distributions import FixedTeamsDistribution


class TestFixedTeams(unittest.TestCase):
    def test_fixed_composition_and_independent_results(self):
        config = {"env_key": "team_gen", "n_units": 32, "n_enemies": 32,
                  "ally_team": {"marine": 20, "marauder": 10, "medivac": 2},
                  "enemy_team": {"marine": 32}}
        distribution = FixedTeamsDistribution(config, np.random.default_rng(1))
        result = distribution.generate()["team_gen"]
        self.assertEqual(result["ally_team"].count("marine"), 20)
        self.assertEqual(len(result["enemy_team"]), 32)
        result["ally_team"].clear()
        self.assertEqual(len(distribution.generate()["team_gen"]["ally_team"]), 32)
        config["n_units"] = 31
        with self.assertRaises(ValueError):
            FixedTeamsDistribution(config, np.random.default_rng(1))
