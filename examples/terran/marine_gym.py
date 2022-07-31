import numpy as np
import gym
from gym import spaces

from enum import Enum
#from stable_baselines import DQN


class GoLeftEnv(gym.Env):
    """Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    # metadata = {'render.modes': ['console']}
    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1

    def __init__(self, grid_size=10):
        super(GoLeftEnv, self).__init__()
        # Size of the 1D-grid
        self.grid_size = grid_size

        # Initialize the agent at the right of the grid
        self.agent_pos = grid_size - 1

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)

        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                  shape=(1,), dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent at the right of the grid
        self.agent_pos = self.grid_size - 1
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32)

    def step(self, action):
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        done = bool(self.agent_pos == 0)

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if self.agent_pos == 0 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([self.agent_pos]).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("." * self.agent_pos, end="")
        print("x", end="")
        print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass


class MarineActionsEnum(Enum):
    ATTACK_CLOSEST_TO_ME = 1
    RUN_AWAY = 2
    ATTACK_LOWEST_LIFE_ENEMY = 3


class MockAgent:
    def __init__(self):
        pass
        #self.dqn = DQN()

    @staticmethod
    def act(observations):
        print(f'got observations - {observations}')
        action = MarineActionsEnum.ATTACK_LOWEST_LIFE_ENEMY
        print(f'acting {action}')
        return action


class MockGame:
    def __init__(self, starting_obs):
        self.observations = starting_obs

    def mock_obs_callback(self, action):
        print(f'game got action {action}')
        if action == MarineActionsEnum.ATTACK_CLOSEST_TO_ME:
            self.observations = np.array(
                [np.max([0, marine_health-5]) for marine_health in self.observations]
            ).astype(int)
        else:
            self.observations = np.append(
                self.observations[0: int(len(self.observations)/2)],
                np.array(
                    [np.max([0, marine_health-5]) for marine_health in self.observations[int(len(self.observations)/2):]]
                ).astype(int)
            )
        print(f'game returned observations')
        print(f'marine mock health = {self.observations}')
        return self.observations


def mock_reward_callback(observation_space):
    return np.random.rand(1)


def reward_health_callback(observation_space):
    return np.sum(
        observation_space[ 0: int(len(observation_space)/2)]
    ) - np.sum(
        observation_space[int(len(observation_space)/2):]
    )


def calc_value_func(marine_health_list):
    """
    my sum of marine health minus the sum of enemy marine's health
    """
    return np.sum(
        marine_health_list[0: int(len(marine_health_list)/2)]
    ) - np.sum(
        marine_health_list[int(len(marine_health_list)/2):]
    )


class MarineGym(gym.Env):
    """Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self, grid_size=16,
                 act_callback=MockAgent.act,
                 reward_callback=mock_reward_callback):

        super(MarineGym, self).__init__()

        # Size of the 1D-grid
        self.grid_size = grid_size

        self.act_callback = act_callback
        self.reward_callback = reward_callback

        # # Initialize the agent at the right of the grid
        self.agent_pos = np.array([45]*self.grid_size).astype(int)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = len(MarineActionsEnum)
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the health of the marines
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=45,
                                            shape=(grid_size,), dtype=int)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # # Initialize the agent at the right of the grid
        # self.agent_pos = self.grid_size - 1
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([45]*self.grid_size).astype(int)

    def update_observations(self, observations):
        self.agent_pos = observations

    def step(self, observations):
        # self._act(action)
        # if action == self.LEFT:
        #     self.agent_pos -= 1
        # elif action == self.RIGHT:
        #     self.agent_pos += 1
        # else:
        #     raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        self.update_observations(observations)
        # # Account for the boundaries of the grid
        # self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        action = self.act_callback(observations=self.agent_pos)

        # the agent doesn't decide when it is done
        # TODO: not sure this is right
        done = False

        reward = self.reward_callback(self.agent_pos)

        # # Are we at the left of the grid?
        # done = bool(self.agent_pos == 0)
        #
        # # Null reward everywhere except when reaching the goal (left of the grid)
        # reward = 1 if self.agent_pos == 0 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return self.agent_pos, action, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print('render!')
        # print("." * self.agent_pos, end="")
        # print("x", end="")
        # print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass


def test_marine_gym_env():
    env = MarineGym(
        grid_size=8,
        act_callback=MockAgent.act,
        reward_callback=reward_health_callback #mock_reward_callback
    )

    obs = env.reset()
    env.render()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    game = MockGame(starting_obs=obs)

    # Hardcoded best agent: always go left!
    n_steps = 20
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        obs, action, reward, done, info = env.step(observations=game.observations)
        print('obs=', obs, 'action=', action,  'reward=', reward, 'done=', done)
        env.render()
        if done:
            print("Goal reached!", "reward=", reward)
            break
        else:
            obs = game.mock_obs_callback(action=action)
            env.update_observations(obs)


if __name__ == '__main__':
    test_marine_gym_env()