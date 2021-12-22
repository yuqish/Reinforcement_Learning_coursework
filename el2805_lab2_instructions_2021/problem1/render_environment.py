import torch
import DQN_network
import gym

### CREATE RL ENVIRONMENT ###
env = gym.make('LunarLander-v2')        # Create a CartPole environment
n = len(env.observation_space.low)   # State space dimensionality
m = env.action_space.n               # Number of actions

nn = torch.load('neural-network-1.pth')


### PLAY ENVIRONMENT ###
# The next while loop plays 5 episode of the environment
for episode in range(5):
    state = env.reset()                  # Reset environment, returns initial
                                         # state
    done = False                         # Boolean variable used to indicate if
                                         # an episode terminated

    while not done:
        env.render()                     # Render the environment
                                         # (DO NOT USE during training of the
                                         # labs...)
        #action  = np.random.randint(m)   # Pick a random integer between
                                         # [0, m-1]
        state_tensor = torch.tensor([state], requires_grad = False)
        x = nn(state_tensor)
        action = x.max(1)[1]
        action = action.item()

        # The next line takes permits you to take an action in the RL environment
        # env.step(action) returns 4 variables:
        # (1) next state; (2) reward; (3) done variable; (4) additional stuff
        next_state, reward, done, _ = env.step(action)

        state = next_state

# Close all the windows
env.close()