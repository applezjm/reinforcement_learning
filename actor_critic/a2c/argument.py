#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
REWARD_BOUND = 18
LEARNING_RATE = 7e-4
OPTIM_EPSILON = 1e-5

GAMMA = 0.99
LOSS_VALUE_COEF = 0.5
LOSS_ENTROPY_COEF = 0.01
GAE_LAMBDA = 0.95
MAX_GRAD_NORM = 0.5

NUM_PROCESSES = 32
NUM_STEPS = 10

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--reward-bound", type=float, default=REWARD_BOUND)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name", default="tmp", help="for board name")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--optim-eps", type=float, default=OPTIM_EPSILON,
                        help="optimizer epsilon")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help="discount factor for rewards")
    parser.add_argument("--value-coef", type=float, default=LOSS_VALUE_COEF,
                        help="coef for value loss")
    parser.add_argument("--entropy-coef", type=float, default=LOSS_ENTROPY_COEF,
                        help="coef for entropy loss")
    parser.add_argument("--gae", action="store_true", default=True)
    parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--log-dir", default="/tmp")
    parser.add_argument("--num-processes", type=int, default=NUM_PROCESSES)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--cuda", default="",
                        help="using cuda device")
    parser.add_argument("--batchnorm", action="store_true", default=True)
    args = parser.parse_args()

    args.cuda = "cpu" if not args.cuda else "cuda:%s" % (args.cuda)

    return args
