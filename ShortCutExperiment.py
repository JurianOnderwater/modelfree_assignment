import numpy as np
from tqdm import tqdm
from ShortCutEnvironment import Environment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from Helper import LearningCurvePlot, ComparisonPlot, smooth


def experiment(n_episodes, n_repetitions, smoothing_window, experiment_type):
    if experiment_type == 1:
        for _ in range(n_repetitions):
            q_learning = QLearningAgent()                                    # initialise the policy
            env = Environment()                                              # initialise the environment
            for _ in range(n_episodes):
                env.reset()                                                 # start with a clean environment
                while not env.done():                                       # continue till you reach terminal state
                    sample_action = q_learning.select_action(env.state())
                    sample_reward = env.step(sample_action)
                    q_learning.update(state=env.state(), action=sample_action, reward=sample_reward)
                    # env.step() # I'm think this is not necessary because .step has the updating of the state as a side effect

    if experiment_type == 2:
        for _ in range(n_repetitions):
            sarsa = SARSAAgent()                                            # initialise the policy
            env = Environment()                                             # initialise the environment
            for _ in range(n_episodes):
                env.reset()                                                 # start with a clean environment
                while not env.done():                                       # continue till you reach terminal state
                    sample_action = sarsa.select_action(env.state())
                    sample_reward = env.step(sample_action)
                    sarsa.update(state=env.state(), action=sample_action, reward=sample_reward)
                    # env.step() # I'm think this is not necessary because .step has the updating of the state as a side effect

    if experiment_type == 3:
        for _ in range(n_repetitions):
            esarsa = ExpectedSARSAAgent()                                            # initialise the policy
            env = Environment()                                             # initialise the environment
            for _ in range(n_episodes):
                env.reset()                                                 # start with a clean environment
                while not env.done():                                       # continue till you reach terminal state
                    sample_action = esarsa.select_action(env.state())
                    sample_reward = env.step(sample_action)
                    esarsa.update(state=env.state(), action=sample_action, reward=sample_reward)
                    # env.step() # I'm think this is not necessary because .step has the updating of the state as a side effect

    pass

if __name__ == '__main__':
    # experiment settings
    n_repetitions       = 500
    n_episodes          = 10000
    n_timesteps         = 1000
    smoothing_window    = 31
    # epsilon_list = [0.01, 0.05, 0.10, 0.25]         #
    # initial_mean_list = [0.1, 0.5, 1.0, 2.0]        # Values that are used in the comparison experiment
    # c_list = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]   #
    # comparison_list = [epsilon_list, initial_mean_list, c_list]
    # experiment_parameter_dict   = {1:'epsilon', 2:'initial_mean', 3:'c', 4:'Comparison'}
    experiment_name_dict        = {1:'Q-learning', 2:'SARSA', 3:'Expected SARSA'}
    experiment_type = int(input('Choose one of Q-learning (1), SARSA (2) or Expected SARSA (3): '))
    # parameter_name = experiment_parameter_dict[experiment_type]

    # if (experiment_type < 4):
    #     print("Add {values}s you want to try out in the {policy}-policy and press 'enter'. When you are done with adding {values}s, press 'enter' again to start the algorithm".format(values=experiment_parameter_dict[experiment_type], policy=experiment_name_dict[experiment_type]))
    #     try:
    #         exploration_parameter_values = ()
    #         while True:
    #             exploration_parameter_values += (float(input('Enter a(n) {parameter_name} and press enter: '.format(parameter_name=parameter_name))),)
    #     except: # if the input is not-integer, just print the list
    #         print(exploration_parameter_values)
    # else:
    #     print('The program will loop over {amount_of_values} values now'.format(amount_of_values=len(epsilon_list)+len(initial_mean_list)+len(c_list)))
    
    experiment(n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window, value=0.1)