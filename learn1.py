from stable_baselines3.common.env_checker import check_env
from ball_balance_env1 import BallBalanceEnv
from stable_baselines3 import PPO

# Initialize your environment
env = BallBalanceEnv(render_mode="rgb_array")

# Check your custom environment
check_env(env)


# Define and train the model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_ball_balance_tensorboard/")
model.learn(total_timesteps=200000, log_interval=4)
model.save("ppo_ball_balance")

# %tensorboard --logdir ./sac_ball_balance_tensorboard/ --port 6006