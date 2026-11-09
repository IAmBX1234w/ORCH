import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from crew_algorithms.wildfire_alg.core.alg_utils import (
    Action, check_if_option_done, generate_action_from_option,
)
from crew_algorithms.wildfire_alg.algorithms.ORCH.agent import Agent


def worker(agent_type=2, option_type=1, target=(52, 50)):
    return SimpleNamespace(
        id=1, type=agent_type, last_position=(51, 49),
        cfg=SimpleNamespace(envs=SimpleNamespace(map_size=100)),
        options=[Action(type=option_type, param_1=target[0], param_2=target[1],
                        description="test option")],
        action_queue=[], past_options=[], extra_variables=[0, 0, 0],
    )


class OptionExecutionTests(unittest.TestCase):
    def test_action_libraries_import_without_credentials_or_model_client(self):
        code = '''
import os, sys
from unittest.mock import patch
for name in list(os.environ):
    if name.endswith('API_KEY'):
        del os.environ[name]
with patch('openai.OpenAI.__init__', side_effect=AssertionError('Unexpected model client')):
    from crew_algorithms.wildfire_alg.libraries import (
        drone_option_library, firefighter_option_library,
        bulldozer_option_library, helicopter_option_library,
    )
assert 'crew_algorithms.wildfire_alg.core.gpt' not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, cwd=Path(__file__).resolve().parents[1], timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_moves_reach_environment_for_every_worker_type(self):
        for agent_type in range(4):
            with self.subTest(agent_type=agent_type):
                agent = worker(agent_type)
                self.assertFalse(check_if_option_done(agent))
                self.assertEqual(generate_action_from_option(agent), [0, 52, 50])
                self.assertFalse(agent._error_idle)
                self.assertEqual(len(agent.options), 1)
                agent.last_position = (52, 50)
                self.assertTrue(check_if_option_done(agent))
                self.assertEqual(generate_action_from_option(agent), [0, 52, 50])
                self.assertEqual(agent.options, [])

    def test_explicit_idle_holds_position_without_error(self):
        agent = worker(option_type=0, target=(0, 0))
        self.assertEqual(generate_action_from_option(agent), [0, 51, 49])
        self.assertFalse(agent._error_idle)

    def test_invalid_action_reports_actual_cause(self):
        agent = worker(target=(-1, 50))
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            self.assertEqual(generate_action_from_option(agent), [0, 51, 49])
        self.assertTrue(agent._error_idle)
        self.assertIn("ValueError: out of bounds target", log.getvalue())

    def test_action_conversion_does_not_swallow_interrupts(self):
        agent = worker()
        with patch('crew_algorithms.wildfire_alg.libraries.drone_option_library.Run_Drone_Option',
                   side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                generate_action_from_option(agent)

    def test_planner_and_capabilities_load_outside_project_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            try:
                os.chdir(elsewhere)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    agent = Agent(id=1, name="AGENT_1", cfg=SimpleNamespace(),
                                  path=elsewhere, api_key="unused", team_name="Drones", type=2)
                self.assertNotIn("Warning:", output.getvalue())
                self.assertIn("Move in the direction", agent.planner)
                manager = SimpleNamespace(children=[agent])
                summary = Agent._build_team_summary(manager)
                self.assertIn("Drone agents are specialized", summary)
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
