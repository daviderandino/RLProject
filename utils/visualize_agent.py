import gymnasium as gym
from stable_baselines3 import PPO
from env.custom_hopper import *


def visualize(model_path, env_id):
    print(f"Caricamento modello da: {model_path}")
    print(f"Visualizzazione su ambiente: {env_id}")

    env = gym.make(env_id, render_mode='human')

    # carico modello addestrato
    model = PPO.load(model_path)

    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"--- Episodio {ep+1} ---")
        
        while not done:
            # deterministic=True perchè non voglio che il modello esplori in training
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            done = terminated or truncated

        print(f"Reward Totale: {total_reward:.2f}")

    env.close()

if __name__ == '__main__':
    
    visualize(
        model_path="models/ppo_source_adr", # modello allenato
        env_id="CustomHopper-target-v0" # ambiente reale
    )
