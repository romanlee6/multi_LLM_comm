from gym_dragon.core import Region
from gym_dragon.envs import DragonEnv



def random_walk(num_episodes=1, num_timesteps=900, regions=set(Region)):
    """
    Visualize a trajectory where agents are taking random actions.
    """
    env = DragonEnv(mission_length=num_timesteps, valid_regions=regions)

    for episode in range(num_episodes):
        obs = env.reset()
        done = {agent_id: False for agent_id in env.get_agent_ids()}
        while not all(done.values()):
            env.render()
            random_action = env.action_space_sample()
            obs, reward, done, info = env.step(random_action)

        print('Episode:', episode, 'Score:', env.score)



if __name__ == '__main__':
    random_walk()
    #random_walk(regions=[Region.village])
