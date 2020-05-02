#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import collections

class ExperienceBuffer:
    def __init__(self, max_len):
        self.buffer = collections.deque(maxlen=max_len)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def store(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, d, s_ = zip(*[self.buffer[idx] for idx in indices])
        return np.array(s), np.array(a), np.array(r), \
            np.array(d), np.array(s_)
