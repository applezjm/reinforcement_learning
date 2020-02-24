#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from conf import *


class Windy_world():
    def stochastic_move(self, state):
        randnum = np.random.rand()
        x, y = state
        if randnum <= 1 / 3:
            return [max(x - 1, 0), y]
        elif randnum <= 2 / 3:
            return [min(x + 1, WORLD_HEIGHT - 1), y]
        else:
            return [x, y]

    def step(self, state, action):
        x, y = state
        if action == ACTION_UP:
            return self.stochastic_move([max(x - 1 - WIND[y], 0), y])
        elif action == ACTION_DOWN:
            return self.stochastic_move([max(min(x + 1 - WIND[y], WORLD_HEIGHT - 1), 0), y])
        elif action == ACTION_LEFT:
            return self.stochastic_move([max(x - WIND[y], 0), max(y - 1, 0)])
        elif action == ACTION_RIGHT:
            return self.stochastic_move([max(x - WIND[y], 0), min(y + 1, WORLD_LENGTH - 1)])
        else:
            assert "ERROR ACTION"
