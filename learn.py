from stable_baselines3.common.env_checker import check_env
from ball_balance_env import BallBalanceEnv
from stable_baselines3 import SAC

# Initialize your environment
env = BallBalanceEnv(render_mode="rgb_array")

# Check your custom environment
check_env(env)


# Define and train the model
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_ball_balance_tensorboard/")
model.learn(total_timesteps=150000, log_interval=4)
model.save("sac_ball_balance")

# %tensorboard --logdir ./sac_ball_balance_tensorboard/ --port 6006