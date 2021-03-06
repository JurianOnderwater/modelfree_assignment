from multiprocessing.sharedctypes import Value
from turtle import delay
import numpy as np
from tqdm import tqdm
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from Helper import LearningCurvePlot, ComparisonPlot, smooth, cumulative_reward, make_averaged_curve, print_greedy_actions
from tqdm import tqdm


def vary_alpha(func):
    '''
    Takes a machine-learning funtion and executes it using a range of values for a hyperparameter in said function.\n
    Generates a plot displaying the performance of the algorithm using the different values.

    Prerequisites:
    ----------
    Function returns the averaged curve\n
    An iterable called 'exploration_parameter_values' should be defined in main, containing the values for the hyperparameter 
    '''
    def wrapper(*args, **kwargs):
        if (experiment_type != 4):
            plot = LearningCurvePlot(title='{title} Experiment'.format(title=experiment_name_dict[experiment_type]))
            for alpha in exploration_parameter_values:
                learning_curve          = func(n_episodes, n_repetitions, experiment_type, alpha)[0]
                smoothed_learning_curve = smooth(learning_curve, smoothing_window)
                plot.add_curve(smoothed_learning_curve, label = 'α = {parameter_value}'.format(parameter_value=alpha))
            plot.save(name = 'varying_α')
        else:
            comparison_plot = ComparisonPlot(title='Policy Comparison')
            optimal_plot = LearningCurvePlot(title='Optimal Parameters')
            for experiment in range(3):
                print('Starting with {experiment} experiment'.format(experiment=experiment_name_dict[experiment+1]))
                rewards = [func(n_episodes, n_repetitions, experiment+1, alpha)[1] for alpha in alpha_list]
                optimal_index = rewards.index(max(rewards))
                optimal = alpha_list[optimal_index]
                optimal_curve = func(n_episodes, n_repetitions, experiment+1, optimal)[0]
                optimal_plot.add_curve(optimal_curve, label = '{exploration_parameter} = {parameter_value}'.format(exploration_parameter=experiment_name_dict[experiment+1],parameter_value=optimal))
                comparison_plot.add_curve(alpha_list, rewards, label='{exploration_parameter}'.format(exploration_parameter=experiment_name_dict[experiment+1]))
            comparison_plot.save(name='comparison_plot')
            optimal_plot.save(name='optimal_plot')
    return wrapper

@vary_alpha
def experiment(n_episodes, n_repetitions, experiment_type, alpha):
    '''
    Execute a machinelearning experiment
    Parameters
    ----------
    n_episodes: amount of episodes used for training
    n_repetitions: amount of training repetitions that is averaged over
    experiment_type: agent that is used
    alpha: learning rate that the agent uses
     '''
    
    max_reward = 0
    env = ShortcutEnvironment()                                                 # initialise the environment
    averaged_curve = [0 for _ in range(n_episodes)]
    print('Starting with α = {alpha}'.format(alpha=alpha))

    for i in tqdm(range(n_repetitions), colour='green'):
        agent_dict = {1:        QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma),
                      2:            SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma), 
                      3:    ExpectedSARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma)}
        agent = agent_dict[experiment_type]
        for j in range(n_episodes):
            env.reset()                                                         # start with a clean environment
            timestep = 0                                                        # counter for average reward
            c_reward = 0                                                        # cumulative reward
            while not env.done():                                               # continue till you reach terminal state                                           
                sample_action = agent.select_action(env.state())                # choose an action to take in the current state
                current_state = env.state()                                     # make a copy of the current state to use in the update function
                sample_reward = env.step(sample_action)                         # get the reward that the taking of the sampled action gives
                agent.update(current_state=current_state, new_state=env.state(), action=sample_action, reward=sample_reward) # update the means that the agent uses to choose an action
                c_reward += cumulative_reward(sample_reward, gamma, timestep)
                timestep += 1 
                if timestep > 3000:                                             # protection against walking too long without final state
                    break
            make_averaged_curve(averaged_curve, c_reward, i, j)                 # update averaged_curve with cumulative reward
    # print_greedy_actions(agent.Q)
    max_reward = averaged_curve[-1]
    return [averaged_curve, max_reward]


pass

if __name__ == '__main__':
    # experiment settings
    n_repetitions       = 100
    n_episodes          = 1000
    smoothing_window    = 31
    epsilon             = 0.1
    gamma               = 1 

    alpha_list = [0.01, 0.1, 0.5, 0.9]        # Values that are used in the comparison experiment
    
    experiment_name_dict        = {1:'Q-learning', 2:'SARSA', 3:'Expected SARSA'}
    experiment_type = int(input('Choose one of Q-learning (1), SARSA (2), Expected SARSA (3) or Comparison (4): '))

    if (experiment_type < 4):
        print("Add alphas you want to try out in the {policy}-policy and press 'enter'. When you are done with adding values, press 'enter' again to start the algorithm".format(policy=experiment_name_dict[experiment_type]))
        try:
            exploration_parameter_values = ()
            while True:
                exploration_parameter_values += (float(input('Enter an α and press enter: ')),)
        except: # if the input is not-integer, just print the list
            print(exploration_parameter_values)
    else:
        print('The program will loop over {amount_of_values} values now for each of the agents'.format(amount_of_values=len(alpha_list)))
    
    experiment(n_repetitions=n_repetitions, n_episodes=n_episodes, experiment_type=experiment_type, alpha=0.1)