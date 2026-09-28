from types import SimpleNamespace

import pytest
from pysc2.lib import protocol

from smac_pettingzoo.env.smacv2 import SC2EpisodeInterrupted, SMACv2EnvCore


def test_reset_keeps_sc2_output_visible(capsys):
    env = SMACv2EnvCore.__new__(SMACv2EnvCore)
    env._episode_count = 0
    env._sc2_proc = None
    env.seed = lambda seed: None

    def fail_launch():
        print("SC2 launch diagnostic")
        raise protocol.ConnectionError("connection lost")

    env._launch = fail_launch
    with pytest.raises(protocol.ConnectionError):
        env.reset(seed=42)
    assert "SC2 launch diagnostic" in capsys.readouterr().out


def test_reset_reuses_a_running_sc2_even_before_first_episode_ends():
    env = SMACv2EnvCore.__new__(SMACv2EnvCore)
    env._episode_count = 0
    env._sc2_proc = object()
    env.seed = lambda seed: None

    def restart():
        raise protocol.ConnectionError("existing process restarted")

    env._restart = restart
    env._launch = lambda: pytest.fail("reset launched a second SC2 process")
    with pytest.raises(protocol.ConnectionError, match="existing process restarted"):
        env.reset(seed=42)


def test_step_disconnect_does_not_fabricate_terminal_transition():
    env = SMACv2EnvCore.__new__(SMACv2EnvCore)
    env.n_agents = 1
    env.n_actions = 2
    env.heuristic_ai = False
    env.debug = False
    env.conic_fov = False
    env.get_agent_action = lambda agent, action: None
    env._controller = SimpleNamespace(
        actions=lambda request: (_ for _ in ()).throw(protocol.ConnectionError("lost"))
    )
    with pytest.raises(SC2EpisodeInterrupted):
        env.step([1])


def test_unit_creation_has_a_finite_limit():
    env = SMACv2EnvCore.__new__(SMACv2EnvCore)
    env.n_agents = env.n_enemies = 1
    env._episode_count = 1
    env.debug = False
    observation = SimpleNamespace(observation=SimpleNamespace(raw_data=SimpleNamespace(units=[])))
    env._obs = observation
    steps = []
    env._controller = SimpleNamespace(
        step=lambda amount: steps.append(amount), observe=lambda: observation
    )
    with pytest.raises(SC2EpisodeInterrupted, match="did not create all units"):
        env.init_units(None, None)
    assert len(steps) == 200
