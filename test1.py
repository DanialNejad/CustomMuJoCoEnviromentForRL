from stable_baselines3 import PPO
from ball_balance_env1 import BallBalanceEnv
import imageio

env = BallBalanceEnv(render_mode="rgb_array")
model = PPO.load("ppo_ball_balance.zip")

obs, info = env.reset()
frames = []
for _ in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    image = env.render()
    if _ % 5 == 0:
        frames.append(image)
    if done or truncated:
        obs, info = env.reset()

imageio.mimsave('ppo_ball_balance1.gif', frames, fps=20)
