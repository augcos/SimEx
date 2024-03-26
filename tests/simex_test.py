import sys
sys.path.insert(0,".")

from stable_baselines3.common.env_checker import check_env
from simex.simex import SimExDiscrete, SimExContinuous


def test_gym_env():
    discrete_exchange_test = SimExDiscrete()
    check_env(discrete_exchange_test)

    continuous_exchange_test = SimExDiscrete()
    check_env(continuous_exchange_test)