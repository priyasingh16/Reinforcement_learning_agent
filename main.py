from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from skimage.transform import resize
from collections import deque
from Utils import Utils
import gym_super_mario_bros
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt



class Agent:


    def __init__(self, height, width, env_name='SuperMarioBros-v0'):
        self.env = gym_super_mario_bros.make(env_name)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.num_actions = self.env.action_space.n
        self.obs = deque(maxlen=4)
        self.height = height
        self.width = width

        # intitial the state with 4 empty frames
        self.obs.append(np.zeros((height, width)))
        self.obs.append(np.zeros((height, width)))
        self.obs.append(np.zeros((height, width)))
        self.obs.append(np.zeros((height, width)))
        self.env.reset()

    def randomAction(self):
        return random.randint(0, self.num_actions-1)


    def play(self, act, curr_time, skip=4):
        current_state = self.obs.copy()
        current_state = np.array(current_state)
        current_state = current_state.transpose(1,2,0)
        r = 0
        for _ in range(0,skip):
            state, reward, done, info = self.env.step(act)
            r = r + reward
            if done or info['time'] <= 1 or info['time'] > curr_time:
                r = r + (-100)
                done = True
                break
            curr_time = info['time']

        state = resize(Utils.preprocess(state), (self.height, self.width), anti_aliasing=True)

        self.obs.append(state)
        next_state = self.obs.copy()
        next_state = np.array(next_state)
        next_state = next_state.transpose(1,2,0)
        return current_state, next_state, r, act, done, curr_time

def main():

        img_height = int(224/2)
        img_width = int(256/2)
        input_shape = (img_height, img_width, 4)
        agent = Agent(img_height, img_width)
        output_shape = agent.num_actions
        agent.env.reset()
        agent.env.close()


if __name__ == "__main__":
    main()
