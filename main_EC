# ==============================
# Simplified Load Classes
# ==============================
# (Replace these with your actual implementations if available.)
from InterruptibleLoad import InterruptibleLoad
from ShiftableLoad import ShiftableLoad
from ContinuouslyAdjustableLoad import ContinuouslyAdjustableLoad
from LoadAdjustmentInStages import LoadAdjustmentInStages

import numpy as np
import random
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env


class EnhancedCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EnhancedCallback, self).__init__(verbose)
        self.episode_rewards = []  # Track rewards for each episode
        self.timesteps = []        # Track timesteps for each episode
        self.current_rewards = 0   # Track cumulative reward within an episode
        self.policy_losses = []    # Track policy loss during training
        self.value_losses = []     # Track value loss during training

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.current_rewards += reward
        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self.current_rewards)
            self.timesteps.append(self.num_timesteps)
            print(f"Episode Reward: {self.current_rewards}")
            self.current_rewards = 0
        return True

    def _on_rollout_end(self):
        logs = self.model.logger.name_to_value
        value_loss = logs.get('train/value_loss')
        policy_loss = logs.get('train/policy_loss')
        if value_loss is not None:
            self.value_losses.append(value_loss)
        else:
            print("[Warning] Value loss not found in the logs.")
        if policy_loss is not None:
            self.policy_losses.append(policy_loss)

    def plot(self):
        plt.figure(figsize=(12, 6))
        # Plot episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.timesteps, self.episode_rewards, label='Episode Rewards', color='blue')
        plt.xlabel('Timesteps')
        plt.ylabel('Total Episode Reward')
        plt.title('Episode Rewards Over Time')
        plt.axhline(y=np.mean(self.episode_rewards), color='red', linestyle='--', label='Average Reward')
        plt.legend()
        plt.grid(True)
        # Plot losses if available
        if len(self.policy_losses) > 0 or len(self.value_losses) > 0:
            plt.subplot(1, 2, 2)
            if len(self.policy_losses) > 0:
                plt.plot(self.policy_losses, label='Policy Loss', color='green')
            if len(self.value_losses) > 0:
                plt.plot(self.value_losses, label='Value Loss', color='orange')
            plt.xlabel('Rollouts')
            plt.ylabel('Loss')
            plt.title('PPO Policy and Value Loss Over Time')
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()

class LoadManagerMDP(gym.Env):
    def __init__(self, profiles, extra_demand, price_per_kW):
        super(LoadManagerMDP, self).__init__()
        self.profiles = profiles
        self.extra_demand = extra_demand
        self.price_per_kW = price_per_kW
        self.total_flexibility_used = 0
        self.total_cost = 0.0
        self.flexibility_contribution = {}
        self.flexibility_costs = {}
        self.loads = {}
        self.current_step = 0
        self.max_steps = 24  # Maximum steps per episode

        # Define action/observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(len(profiles),), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=50, shape=(10,), dtype=np.float32)

        # Generate random flexibility limits for each profile
        self.flexibility_limits = {profile: random.uniform(1000, 2000) for profile in profiles.keys()}

        self._create_loads()

    def _create_loads(self):
        profiles_to_loads = {
            'Profile 1': (self.profiles['Profile 1'], InterruptibleLoad, {'max_power_reduction': 200}),
            'Profile 2': (self.profiles['Profile 2'], ShiftableLoad, {'max_power_profile': self.profiles['Profile 2'], 't_run': 4, 't_end': 24, 't_req': 3, 'ramping_limit': 150}),
            'Profile 3': (self.profiles['Profile 3'], ContinuouslyAdjustableLoad, {'max_power_change': 50}),
            'Profile 4': (self.profiles['Profile 4'], LoadAdjustmentInStages, {'ramping_limit': 200}),
            'Profile 5': (self.profiles['Profile 5'], InterruptibleLoad, {'max_power_reduction': 500}),
            'Profile 6': (self.profiles['Profile 6'], ShiftableLoad, {'max_power_profile': self.profiles['Profile 6'], 't_run': 4, 't_end': 24, 't_req': 3, 'ramping_limit': 300}),
            'Profile 7': (self.profiles['Profile 7'], ContinuouslyAdjustableLoad, {'max_power_change': 50}),
            'Profile 8': (self.profiles['Profile 8'], LoadAdjustmentInStages, {'ramping_limit': 200}),
            'Profile 9': (self.profiles['Profile 9'], InterruptibleLoad, {'max_power_reduction': 800}),
            'Profile 10': (self.profiles['Profile 10'], ShiftableLoad, {'max_power_profile': self.profiles['Profile 10'], 't_run': 4, 't_end': 24, 't_req': 3, 'ramping_limit': 100}),
        }

        for load_id, (profile_data, LoadType, params) in profiles_to_loads.items():
            load = LoadType(
                profile=profile_data,
                flexibility_fraction=0.1,
                cost_function=lambda x: x * self.price_per_kW,
                load_id=load_id,
                **params
            )
            self.loads[load_id] = load

        self.flexibility_contribution = {ld: 0.0 for ld in self.loads}
        self.flexibility_costs = {ld: 0.0 for ld in self.loads}

    def reset(self, seed=None, options=None):
        self.total_flexibility_used = 0
        self.total_cost = 0.0
        self.current_step = 0
        self.flexibility_contribution = {ld: 0.0 for ld in self.loads}
        self.flexibility_costs = {ld: 0.0 for ld in self.loads}

        state = np.zeros(len(self.loads), dtype=np.float32)
        if len(state) < 10:
            state = np.pad(state, (0, 10 - len(state)), 'constant')
        return state, {}

    def step(self, action):
        profile_flexibilities = np.zeros(len(self.loads))
        total_flexibility = 0.0
        total_cost = 0.0

        for i, (load_id, load) in enumerate(self.loads.items()):
            calc_flex = random.uniform(1000, 2000)
            adj_flex = calc_flex * action[i]
            adj_flex = max(0.1, adj_flex)

            flex_limit = self.flexibility_limits[load_id]
            remain_flex = max(0.0, flex_limit - self.flexibility_contribution[load_id])
            max_allowable = min(adj_flex, remain_flex)
            adj_flex = min(adj_flex, max_allowable)
            adj_flex = max(0.0, adj_flex)

            profile_flexibilities[i] = adj_flex
            total_flexibility += adj_flex

            profile_cost = load.cost_function(adj_flex)
            total_cost += profile_cost

        if total_flexibility > self.extra_demand:
            scaling_factor = 1.0 + (self.extra_demand - total_flexibility) / total_flexibility
            scaling_factor = np.clip(scaling_factor, 0.9, 1.1)
            profile_flexibilities *= scaling_factor
            total_cost *= scaling_factor
            total_flexibility = self.extra_demand

        self.total_flexibility_used += total_flexibility
        self.total_cost += total_cost

        for i, load_id in enumerate(self.loads.keys()):
            self.flexibility_contribution[load_id] += profile_flexibilities[i]
            cost_val = self.loads[load_id].cost_function(profile_flexibilities[i])
            self.flexibility_costs[load_id] += cost_val

        # Consolidated weighted reward calculation
        total_reward = self.calculate_weighted_reward()

        state = np.array(profile_flexibilities, dtype=np.float32)
        if len(state) < 10:
            state = np.pad(state, (0, 10 - len(state)), 'constant')
        state = np.clip(state, 0.0, 50.0)

        self.current_step += 1
        terminated = bool(self.total_flexibility_used >= self.extra_demand or self.current_step >= self.max_steps)
        truncated = False

        return state, total_reward, terminated, truncated, {}

    def calculate_weighted_reward(self):
        flexibility_weight = 0.6
        cost_weight = 0.3
        # demand_penalty based on gap between required and achieved flexibility.
        demand_penalty = -10 * ((self.extra_demand - self.total_flexibility_used) ** 2) / (self.extra_demand ** 2)
        flexibility_reward = flexibility_weight * (self.total_flexibility_used / self.extra_demand) if self.extra_demand > 0 else 0.0
        cost_penalty = cost_weight * -0.0001 * self.total_cost

        total_reward = demand_penalty + flexibility_reward + cost_penalty
        return np.clip(total_reward, -100, 100)

    def render(self, mode='human'):
        print(f"Total Flexibility Used: {self.total_flexibility_used}")
        for load_id, contrib in self.flexibility_contribution.items():
            print(f"{load_id}: Flexibility Contribution: {contrib} (Limit: {self.flexibility_limits[load_id]})")

class CentralEnergyManager:
    def __init__(self, environments):
        self.environments = environments
        self.total_flexibility = 0.0
        self.total_flexibility_history = []  # Record total flexibility per step

    def distribute_energy(self):
        total_flex = 0.0
        for env in self.environments:
            flex_contribution = env.get_attr('flexibility_contribution')[0]
            env_flex = sum(flex_contribution.values())
            total_flex += env_flex
        self.total_flexibility = total_flex
        self.total_flexibility_history.append(total_flex)
        for env in self.environments:
            current_extra_demand = env.get_attr('extra_demand')[0]
            if self.total_flexibility > 0:
                new_extra_demand = min(current_extra_demand, self.total_flexibility)
                env.set_attr('extra_demand', new_extra_demand)
            else:
                env.set_attr('extra_demand', current_extra_demand)

    def run_simulation(self, model, timesteps=1000):
        obs_list = []
        for env in self.environments:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs, info = reset_result, {}
            obs_list.append((obs, info))
        for _ in range(timesteps):
            self.distribute_energy()
            for i, env in enumerate(self.environments):
                obs, info = obs_list[i]
                action, _ = model.predict(obs)
                result = env.step(action)
                if len(result) == 5:
                    obs, rewards, terminated, truncated, info = result
                else:
                    obs, rewards, terminated, info = result
                    truncated = False
                obs_list[i] = (obs, info)
                if terminated or truncated:
                    reset_result = env.reset()
                    if isinstance(reset_result, tuple):
                        obs, info = reset_result
                    else:
                        obs, info = reset_result, {}
                    obs_list[i] = (obs, info)

def get_total_flexibility_over_time(central_manager):
    return central_manager.total_flexibility_history

def plot_total_flexibility_over_time(total_flexibility_history):
    time_steps = np.arange(len(total_flexibility_history))
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, total_flexibility_history, label='Total Flexibility', color='blue', linewidth=2)
    plt.xlabel('Timesteps')
    plt.ylabel('Total Flexibility (W)')
    plt.title('Total Flexibility Over Time (All Environments)')
    plt.grid(True)
    plt.axhline(y=np.mean(total_flexibility_history), color='red', linestyle='--', label='Avg Flexibility')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_flexibility_contributions(envs):
    all_flex_contrib = []
    all_flex_costs = []
    load_ids = list(envs[0].get_attr('flexibility_contribution')[0].keys())
    num_profiles = len(load_ids)
    for env in envs:
        flex_contrib = env.get_attr('flexibility_contribution')[0]
        flex_costs = env.get_attr('flexibility_costs')[0]
        all_flex_contrib.append([flex_contrib[ld] for ld in load_ids])
        all_flex_costs.append([flex_costs[ld] for ld in load_ids])
    all_flex_contrib = np.array(all_flex_contrib)
    all_flex_costs = np.array(all_flex_costs)
    profiles = np.arange(1, num_profiles + 1)
    plt.figure(figsize=(14, 7))
    for i in range(all_flex_contrib.shape[0]):
        plt.bar(profiles + i * 0.1, all_flex_contrib[i], width=0.1, label=f'Env {i + 1}')
    plt.xlabel('Profiles')
    plt.ylabel('Flexibility Contribution (W)')
    plt.title('Flexibility Contribution per Environment')
    plt.xticks(profiles + 0.1 * (len(envs) - 1) / 2, [f'Profile {p}' for p in profiles])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_flexibility_and_costs(envs, central_manager):
    for i, env in enumerate(envs):
        flex_contribution = env.get_attr('flexibility_contribution')[0]
        flex_costs = env.get_attr('flexibility_costs')[0]
        total_flex = sum(flex_contribution.values())
        total_cost = sum(flex_costs.values())
        print(f"Environment {i + 1} - Total Flexibility: {total_flex:.2f} W")
        print(f"Environment {i + 1} - Total Cost: {total_cost:.2f} Units\n")
    overall_flexibility = central_manager.total_flexibility
    print(f"Overall Total Flexibility Across All Environments: {overall_flexibility:.2f} W")

final_all_flexibility_data = None

def run_simulation_and_get_flex_data():
    """
    This function runs the simulation, trains the PPO model (if not already saved),
    executes the central energy manager simulation and returns the final flexibility data.
    It also saves CSV files containing:
      - The final flexibility contributions and limits for each profile.
      - The training metrics (timesteps, episode rewards, policy loss, and value loss) from the callback.
    """
    # Load profiles from Excel if available; otherwise, use synthetic data.
    excel_file_path = 'Residential-Profiles2023.xlsx'
    if os.path.exists(excel_file_path):
        profiles_week = pd.read_excel(excel_file_path)
        profiles = {
            'Profile 1': profiles_week[['Household 1', 'Household 2', 'Household 3']].sum(axis=1).values / 
                         profiles_week[['Household 1', 'Household 2', 'Household 3']].sum(axis=1).max() * 50,
            'Profile 2': profiles_week[['Household 6', 'Household 7', 'Household 8']].sum(axis=1).values / 
                         profiles_week[['Household 6', 'Household 7', 'Household 8']].sum(axis=1).max() * 50,
            'Profile 3': profiles_week[['Household 11', 'Household 12', 'Household 13']].sum(axis=1).values / 
                         profiles_week[['Household 11', 'Household 12', 'Household 13']].sum(axis=1).max() * 50,
            'Profile 4': profiles_week[['Household 16', 'Household 17', 'Household 18']].sum(axis=1).values / 
                         profiles_week[['Household 16', 'Household 17', 'Household 18']].sum(axis=1).max() * 50,
            'Profile 5': profiles_week[['Household 21', 'Household 22', 'Household 23']].sum(axis=1).values / 
                         profiles_week[['Household 21', 'Household 22', 'Household 23']].sum(axis=1).max() * 50,
            'Profile 6': profiles_week[['Household 26', 'Household 27', 'Household 28']].sum(axis=1).values / 
                         profiles_week[['Household 26', 'Household 27', 'Household 28']].sum(axis=1).max() * 50,
            'Profile 7': profiles_week[['Household 31', 'Household 32', 'Household 33']].sum(axis=1).values / 
                         profiles_week[['Household 31', 'Household 32', 'Household 33']].sum(axis=1).max() * 50,
            'Profile 8': profiles_week[['Household 36', 'Household 37', 'Household 38']].sum(axis=1).values / 
                         profiles_week[['Household 36', 'Household 37', 'Household 38']].sum(axis=1).max() * 50,
            'Profile 9': profiles_week[['Household 41', 'Household 42', 'Household 43']].sum(axis=1).values / 
                         profiles_week[['Household 41', 'Household 42', 'Household 43']].sum(axis=1).max() * 50,
            'Profile 10': profiles_week[['Household 46', 'Household 47', 'Household 48']].sum(axis=1).values / 
                         profiles_week[['Household 46', 'Household 47', 'Household 48']].sum(axis=1).max() * 50,
        }
    else:
        print(f"Excel file '{excel_file_path}' not found. Using synthetic data instead...")
        time_steps = 24
        profiles = {f'Profile {i}': np.random.rand(time_steps) * 50 for i in range(1, 11)}

    prices_per_kW = [0.01, 0.05, 0.07, 0.09, 0.1, 0.02, 0.03, 0.07, 0.06, 0.3]
    num_envs = len(prices_per_kW)

    environments = [
        VecNormalize(
            DummyVecEnv([lambda p=profiles, price=prices_per_kW[i]:
                         LoadManagerMDP(profiles=p, extra_demand=20000, price_per_kW=price)]),
            norm_obs=True,
            norm_reward=False,
            clip_reward=10
        )
        for i in range(num_envs)
    ]

    model_filename = "ppo_energy_manager.zip"
    vecnormalize_filename = "ppo_vecnormalize.pkl"

    # If a saved model exists, load it; otherwise, train.
    if os.path.exists(model_filename) and os.path.exists(vecnormalize_filename):
        print("Loading saved model and VecNormalize state...")
        model = PPO.load(model_filename, env=environments[0])
        environments[0] = VecNormalize.load(vecnormalize_filename, environments[0])
        callback = None  # No new training, so no callback metrics.
    else:
        print("No saved model found. Starting training...")
        model = PPO(
            "MlpPolicy",
            environments[0],
            verbose=1,
            learning_rate=1e-5,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=0.5,
            clip_range=0.2,
            max_grad_norm=0.5,
            ent_coef=0.01
        )
        callback = EnhancedCallback()
        # Check the environment for correctness
        env_check = LoadManagerMDP(profiles, extra_demand=20000, price_per_kW=prices_per_kW[0])
        check_env(env_check)

        print("Starting training...")
        model.learn(total_timesteps=500000, callback=callback)
        print("Training completed.")

        # Save the trained model weights and VecNormalize statistics
        model.save(model_filename)
        environments[0].save(vecnormalize_filename)
        # Reload for consistency
        model = PPO.load(model_filename, env=environments[0])
        environments[0] = VecNormalize.load(vecnormalize_filename, environments[0])
        
        # --- Save training metrics (reward and loss) to a CSV file ---
        # Determine the maximum length among metric lists and pad the shorter ones.
        max_len = max(len(callback.timesteps), len(callback.episode_rewards),
                      len(callback.policy_losses), len(callback.value_losses))
        
        def pad_list(lst, length):
            return lst + [np.nan] * (length - len(lst))
        
        metrics_dict = {
            "Timesteps": pad_list(callback.timesteps, max_len),
            "Episode Rewards": pad_list(callback.episode_rewards, max_len),
            "Policy Loss": pad_list(callback.policy_losses, max_len),
            "Value Loss": pad_list(callback.value_losses, max_len)
        }
        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv("ppo_training_metrics.csv", index=False)
        print("Saved training metrics to ppo_training_metrics.csv")

    central_manager = CentralEnergyManager(environments)
    print("Running central energy manager simulation...")
    central_manager.run_simulation(model, timesteps=1000)
    print("Simulation completed.")

    # Get the final flexibility history data
    final_data = get_total_flexibility_over_time(central_manager)
    print("Final total flexibility data (over time):", final_data)
    plot_total_flexibility_over_time(final_data)
    
    # --- Save final flexibility contributions and limits to a CSV file ---
    # Retrieve the underlying environment from DummyVecEnv
    env0 = environments[0].envs[0]
    flex_contrib = env0.flexibility_contribution  # Already a dictionary
    flex_limits = env0.flexibility_limits
    data_list = []
    for profile, contrib in flex_contrib.items():
        limit = flex_limits.get(profile, np.nan)
        data_list.append({'Profile': profile, 'Flexibility Contribution': contrib, 'Flexibility Limit': limit})
    df_flex = pd.DataFrame(data_list)
    df_flex.to_csv("ppo_flexibility_contributions.csv", index=False)
    print("Saved flexibility contributions to ppo_flexibility_contributions.csv")
    
    return final_data

if __name__ == "__main__":
    # When running this module directly, run the simulation and show the final data.
    final_all_flexibility_data = run_simulation_and_get_flex_data()
    # Additional code can be added here if needed.

