from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from skimage.transform import resize
from collections import deque
from Utils import Utils
from Model import DeepQModel
import gym_super_mario_bros
import numpy as np 
import random
import tensorflow as tf
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, height, width, env_name='SuperMarioBros-v0'):
        # Create gym environment
        self.env = gym_super_mario_bros.make(env_name)

        # Adding actions to the environment
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

        self.num_actions = self.env.action_space.n
        # Define state as a queue
        self.obs = deque(maxlen=4)
        self.height = height
        self.width = width

        # Initialize state with empty frames
        self.obs.append(np.zeros((height, width)))
        self.obs.append(np.zeros((height, width)))
        self.obs.append(np.zeros((height, width)))
        self.obs.append(np.zeros((height, width)))

        self.env.reset()

    def randomAction(self):
        return random.randint(0, self.num_actions-1)

    def play(self, act, curr_time, skip_frame=4):
        
        current_state = self.obs.copy()
        current_state = np.array(current_state)
        current_state = current_state.transpose(1,2,0)

        r = 0
        for _ in range(0, skip_frame):
            state, reward, done, info = self.env.step(act)
            r = r + reward
            if done or info['time'] <= 1 or info['time'] > curr_time:
                r = r + (-100)
                done = True
                break
            curr_time = info['time']

        state = resize(Utils.pre_process(state), (self.height, self.width), anti_aliasing=True)

        self.obs.append(state)
        next_state = self.obs.copy()
        next_state = np.array(next_state)
        next_state = next_state.transpose(1,2,0)
        return current_state, next_state, r, done, curr_time


def main():
    # Defining size of the frame by reducing it by half
    img_height = int(224/2)
    img_width = int(256/2)

    # Define shape of the input - stack of 4 frames
    input_shape = (img_height, img_width, 4)

    # Create agent
    agent = Agent(img_height, img_width)
    output_shape = agent.num_actions
    agent.env.reset()
    agent.env.close()
    model = DeepQModel(input_shape, output_shape, 0.01, 0.999)
    episode = 1000

    for i in range(episode):

        agent = Agent(img_height, img_width)
        current_state = agent.obs.copy()
        current_state = np.array(current_state)
        current_state = current_state.transpose(1,2,0)
        current_state = np.array([current_state])
        curr_time = 400

        for step in range(0, 10000):
            action = agent.randomAction() if model.epsilon_condition() else\
                np.argmax(model.predict([current_state])[0])
            
            current_state, next_state, reward, done, curr_time = agent.play(action, curr_time)
            current_state = np.array([current_state])
            next_state = np.array([next_state])
            model.appendReplay((current_state, action, reward, next_state, done))
            current_state = next_state
            # agent.env.render()
            if done:
                print("Episode", i, step, model.epsilon)
                model.syncNetworks()
                break
            model.train()
            agent.env.reset()
            agent.env.close()


if __name__ == "__main__":
    main()
