#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 世界的长宽
WORLD_HEIGHT = 7
WORLD_LENGTH = 10

# 风力等级
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# 动作
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# 更新速率
LEARNING_RATE = 0.5

# 贪婪法e
EPSILON = 0.1

# 反馈
REWARD = -1.0

# 起止点
START = [3, 0]
END = [3, 7]
