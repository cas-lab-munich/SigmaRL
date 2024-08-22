from utilities.mappo_cavs import mappo_cavs
from utilities.helper_training import Parameters

config_file = "config.json"  # Adjust parameters therein
parameters = Parameters.from_json(config_file)
env, policy, parameters = mappo_cavs(parameters=parameters)