#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Cumlative Reward')      
        # self.ax.set_ylim([0,1.0])
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,y,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y,label=label)
        else:
            self.ax.plot(y)
        
    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

class ComparisonPlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Alpha')
        self.ax.set_ylabel('Cumulative reward in last episode') 
        self.ax.set_xscale('log')
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,x,y,label=None):
        ''' x: vector of parameter values
        y: vector of associated mean reward for the parameter values in x 
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(x,y,label=label)
        else:
            self.ax.plot(x,y)
        
    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def print_greedy_actions(Q):
     greedy_actions = np.argmax(Q, 1).reshape((12,12))
     print_string = np.zeros((12, 12), dtype=str)
     print_string[greedy_actions==0] = '^'
     print_string[greedy_actions==1] = 'v'
     print_string[greedy_actions==2] = '<'
     print_string[greedy_actions==3] = '>'
     print_string[np.max(Q, 1).reshape((12, 12))==0] = ' '
     line_breaks = np.zeros((12,1), dtype=str)
     line_breaks[:] = '\n'
     print_string = np.hstack((print_string, line_breaks))
     print(print_string.tobytes().decode('utf-8'))

def cumulative_reward(reward: float, gamma: float, timestep: int)->float:
    '''
    Returns the cumulative reward for a gamma and timestep.

    Parameters
    ----------
    reward: reward to add to cumulative reward
    gamma: discount factor
    timestep: exponent to apply to gamma
    
    Returns
    ----------
    Amount to add to cumulative reward'''
    return (reward * pow(gamma, timestep))

def make_averaged_curve(averaged_curve: list, value: float, iterator: int, location: int)->None:
    '''
    Parameters
    ----------
    averaged_curve: list containing the values of hte curve
    value: the value that will be added to the running average of averaged_curve[location]
    iterator: iterator over which is averaged
    location: iterator which is used for the index in averaged_curve'''
    try:
        averaged_curve[location] += (1 / iterator) * (value - averaged_curve[location])   #(average learning-curve/reward over n_repetitions) #dont know yet how to do this
    except ZeroDivisionError:
        averaged_curve[location] += value

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

if __name__ == '__main__':
    # Test Learning curve plot
    x = np.arange(100)
    y = 0.01*x + np.random.rand(100) - 0.4 # generate some learning curve y
    LCTest = LearningCurvePlot(title="Test Learning Curve")
    LCTest.add_curve(y,label='method 1')
    LCTest.add_curve(smooth(y,window=35),label='method 1 smoothed')
    LCTest.save(name='learning_curve_test.png')

    # Test Performance plot
    PerfTest = ComparisonPlot(title="Test Comparison")
    PerfTest.add_curve(np.arange(5),np.random.rand(5),label='method 1')
    PerfTest.add_curve(np.arange(5),np.random.rand(5),label='method 2')
    PerfTest.save(name='comparison_test.png') 