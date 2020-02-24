#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from conf import *
from gridworld import Windy_world
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


class Q_LEARNING():
    def __init__(self):
        self.max_episode = 1000
        self.env = Windy_world()
        self.steps = [] 
        # 不构造世界形状，只定义价值函数
        self.q_values = np.zeros((WORLD_HEIGHT, WORLD_LENGTH, len(ACTIONS)))
    
    def e_greedy_choose_action(self, state):
        if np.random.rand() < EPSILON:
            return np.random.choice(ACTIONS)
        else:
            values_ = self.q_values[state[0], state[1], :]
            return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    def epsiode(self):
        # total time step in one epsiode
        step = 0

        # s0, a0
        state = START

        while state != END:
            action = self.e_greedy_choose_action(state)

            # record next step
            next_state = self.env.step(state, action)
            values_ = self.q_values[next_state[0], next_state[1], :]
            next_action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

            # q learning update
            self.q_values[state[0], state[1], action] += LEARNING_RATE * \
                (REWARD + self.q_values[next_state[0], next_state[1], next_action] -
                 self.q_values[state[0], state[1], action])

            # next step
            state = next_state
            step += 1
        
        return step

    def draw(self):
        plt.plot(self.steps, np.arange(1, len(self.steps) + 1))
        plt.xlabel("time steps")
        plt.ylabel("episodes")
        plt.savefig("qlearning.png")
        plt.close()

        optimal_policy = [["" for _ in range(WORLD_LENGTH)] for _ in range(WORLD_HEIGHT)]
        for i in range(WORLD_HEIGHT):
            for j in range(WORLD_LENGTH):
                if [i, j] == END:
                    optimal_policy[i][j] = "E"
                    continue
                best_action  = np.argmax(self.q_values[i, j, :])
                if best_action == ACTION_UP:
                    optimal_policy[i][j] = "U"
                elif best_action == ACTION_DOWN:
                    optimal_policy[i][j] = "D"
                elif best_action == ACTION_LEFT:
                    optimal_policy[i][j] = "L"
                elif best_action == ACTION_RIGHT:
                    optimal_policy[i][j] = "R"
        print("the best optimal policy after 1000 epsiodes is:")
        for row in optimal_policy:
            print(row)
        print([str(x) for x in WIND])
    
    def run(self):
        ep = 0
        while ep < self.max_episode:
            self.steps.append(self.epsiode())
            ep += 1
        self.steps = np.add.accumulate(self.steps)

if __name__ == "__main__":
    q = Q_LEARNING()
    q.run()
    q.draw()











