import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from sdnSyncGameDictEncoded import sdnSync_SP
from sdnSyncGameDictEncoded import sdnSync_LB
from itertools import product
from agents.actor_critic_agents.A2C import A2C
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.A3C import A3C
from agents.policy_gradient_agents.PPO import PPO
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from agents.DQN_agents.DQN import DQN
from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets



config = Config()

config.seed = 10
config.num_episodes_to_run = 50
config.file_to_save_data_results = "results/data_and_graphs/SDN_sync_Results_Data.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/SDN_sync_Results_Graph.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.baselines = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = False
config.save_model = True
config.qExcluder = True
config.NNrestarter = False
config.alternator = True

config.environment = sdnSync_LB(config.alternator, numStates=7, numActions=7, budget=2, place_budget=3, numSteps = 1000) 

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.01,
        "batch_size": 256,
        "buffer_size": 40000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 10,
        "discount_rate": 0.1,  #0.99,
        "tau": 0.01,
        "update_every_n_steps": 1,
        "linear_hidden_units": [50],   #[20,30],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 1,
        "clip_rewards": False
    },
    "Stochastic_Policy_Search_Agents": {
        "policy_network_type": "Linear",
        "noise_scale_start": 1e-2,
        "noise_scale_min": 1e-3,
        "noise_scale_max": 2.0,
        "noise_scale_growth_factor": 2.0,
        "stochastic_action_decision": False,
        "num_policies": 10,
        "episodes_per_policy": 1,
        "num_policies_to_keep": 5,
        "clip_rewards": False
    },
    "Policy_Gradient_Agents": {
        "learning_rate": 0.05,
        "linear_hidden_units": [20, 20],
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 5,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 4,
        "normalise_rewards": True,
        "gradient_clipping_norm": 7.0,
        "mu": 0.0, #only required for continuous action games
        "theta": 0.0, #only required for continuous action games
        "sigma": 0.0, #only required for continuous action games
        "epsilon_decay_rate_denominator": 1.0,
        "clip_rewards": False
    },

    "Actor_Critic_Agents":  {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}

if __name__ == "__main__":


     # Define the hyperparameter search space
    # learning_rates = [0.005, 0.01 , 0.05]
    # hidden_units = [[20],[50],[20,20],[20,30]]
    # batch_size = [128,256,512,1024]
    # buffer_size = [20000,40000,80000]
    # epsilon_decay_rate_denominator = [5,10,20]
    # discount_rate = [0.1,0.2,0.4,0.98,0.99]

    # # Perform grid search
    # for lr, hu, bs, bu, ed, disc_r in product(learning_rates, hidden_units, batch_size, buffer_size, epsilon_decay_rate_denominator,discount_rate):
    #     # Set the hyperparameters
    #     hyperparams = {
    #         "learning_rate": lr,
    #         "linear_hidden_units": hu,
    #         "batch_size": bs,
    #         "buffer_size":bu,
    #         "epsilon_decay_rate_denominator":ed,
    #         "discount_rate":disc_r
        
    #     }

    #     config.hyperparameters["DQN_Agents"]["learning_rate"] = hyperparams["learning_rate"]
    #     config.hyperparameters["DQN_Agents"]["linear_hidden_units"] = hyperparams["linear_hidden_units"]
    #     config.hyperparameters["DQN_Agents"]["batch_size"] = hyperparams["batch_size"]
    #     config.hyperparameters["DQN_Agents"]["buffer_size"] = hyperparams["buffer_size"]
    #     config.hyperparameters["DQN_Agents"]["epsilon_decay_rate_denominator"] = hyperparams["epsilon_decay_rate_denominator"]
    #     config.hyperparameters["DQN_Agents"]["discount_rate"] = hyperparams["discount_rate"]

        AGENTS = [DDQN,DQN,Dueling_DDQN]  #[DDQN]  #[SAC_Discrete, DDQN, Dueling_DDQN, DQN, DQN_With_Fixed_Q_Targets,
                #DDQN_With_Prioritised_Experience_Replay, A2C, PPO, A3C ]
        trainer = Trainer(config, AGENTS)
        trainer.run_games_for_agents()






