from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Tuple
from tqdm import tqdm
import numpy as np
import torch
import numpy as np
import pygame
from deep_sarsa import DeepSarsa
from functools import partial
from typing import Callable
import json
import os
from utils import recursive_remove_key
from agents.policies import SingleAgentAction, SingleAgentPolicy
from agents.robot_integration import RobotActionType, RobotActionRequest
from env_utils import SpaceInvadersState, StepInfo, WelfareInfo, WelfareStepInfo, SpaceInvadersConfig
from utils import CustomJSONEncoder, CustomJSONDecoder
import random


from consts import ObservationConsts, ValueFunctionType, NaoSupportPolicies, Players, RewardConsts, DynamicsConsts, InteractiveModes, ActionSpaces
from agents.control_mapping import PyGameGamepad, PyGameKeyboard

if TYPE_CHECKING:
    from agents.policies import Action, Policy
    from agents.robot_integration import RobotActionQueue
    from utils import EnvRenderWrapper

# TODO: create observation config class to hold things like minimal complexity env and pass around jointly
# TODO: expand that to a q learning config to hold all q learning parameters. use for path management

class QValueFunction:
    def __init__(self, model: 'DeepSarsa', minimal_complexity_env:bool = False, device="cpu"):
        self.model = model
        self.model.q_net.to(device)
        self.minimal_complexity_env = minimal_complexity_env

    def __call__(self, recent_states: Tuple['SpaceInvadersState'], action: 'Action') -> float:

        recent_observations = [state.get_observation(minimal_observation=self.minimal_complexity_env) for state in recent_states]
        stacked_observation = np.concatenate(recent_observations, axis=0)
        expected_obs_dim = self.model.observation_space.shape[0]
        stacked_observation = np.pad(stacked_observation, (expected_obs_dim - stacked_observation.size, 0), mode='constant', constant_values=-1)
        obs = stacked_observation

        q_values = (
            self.model.q_net(torch.tensor(obs).reshape(1, -1))
            .detach()
            .cpu()
            .numpy()
        )[0]
        value = q_values[action.index()]
        return value

class ValueFunction:
    def __init__(self, model: 'DeepSarsa', expected_agent_policy: 'Policy', minimal_complexity_env:bool = False):
        self.type = ValueFunctionType.Q_FUNCTION
        self.discount_factor = model.gamma
        self.q_value_function = QValueFunction(
            model=model, 
            minimal_complexity_env=minimal_complexity_env,
        )
        self.expected_agent_policy = expected_agent_policy

    def __call__(self, recent_states: Tuple['SpaceInvadersState', ...]) -> float:
        current_state = recent_states[-1]
        expected_agent_action = self.expected_agent_policy(current_state)
        value = self.q_value_function(recent_states, expected_agent_action)
        return value

class TrajRollout_ValueFunction:
    def __init__(self, gamma: float, env: 'EnvRenderWrapper', expected_agent_policy: 'Policy', num_rollouts: int):
        self.type = ValueFunctionType.MC_ROLLOUT
        self.discount_factor = gamma
        self.expected_agent_policy = expected_agent_policy
        self.env = env
        self.num_rollouts = num_rollouts
    
    def __call__(self, recent_states: Tuple['SpaceInvadersState', ...]) -> float:
        # Monte-Carlo estimate of the value of a state
        state = recent_states[-1]
       
        values = []
        for i in range(self.num_rollouts):
            traj = rollout_trajectory(env=self.env, agent_policy=self.expected_agent_policy, start_state=state)
            discounted_future_reward = 0
            for i, step in enumerate(traj):
                discounted_future_reward += step.reward * (self.discount_factor ** i)
            values.append(discounted_future_reward)
        value = sum(values) / len(values)
        
        return value
    
    # def __call__(self, state: 'SpaceInvadersState', n_trajs=100) -> float:
    #     # Monte-Carlo estimate of the value of a state
    #     # Recall that the value of a state is the expected sum of rewards from that state
    #     trajs = [rollout_trajectory(env=self.env, agent_policy=self.expected_agent_policy, state=state) for _ in range(n_trajs)]
    #     discounted_future_rewards = [get_discounted_future_rewards(traj, self.discount_factor) for traj in trajs]
    #     values = [sum(dfr) for dfr in discounted_future_rewards]
    #     value = sum(values) / len(values)
    #     return value


# Seed setting code taken from stable_baselines3.common.utils.set_random_seed and stable_baselines3.common.base_class.load
def load_model(model_path: str, seed:int = None) -> 'DeepSarsa':
    model = DeepSarsa.load(model_path)
    model.seed = seed
    model._setup_model()
    if model.use_sde:
        model.policy.reset_noise()

    return model

# Get human expectations as a value function
# def get_human_expectations_value_function(
#         game_type: 'GameTypes',
#         human_policy_type: 'HumanPolicies', 
#         expected_support_policy: 'NaoSupportPolicies', 
#         reward_model: 'RewardTypes', 
#         model_seed: int=None,
#         model_path_override: str=None,
#         expected_agent_policy: 'Policy',
# ) -> ValueFunction:
    
#     if model_path_override is not None:
#         _exp_dir, model_path = get_model_path(game_type, human_policy_type, expected_support_policy, reward_model)
#     else:
#         model_path = model_path_override
#     expectations_model = load_model(model_path, model_seed)
#     value_function = ValueFunction(expectations_model, expected_agent_policy)
#     return value_function

    
def step_from_state(env: 'EnvRenderWrapper', state: 'SpaceInvadersState', action: 'Action') -> Tuple['SpaceInvadersState', float]:
    env.set_state(state)
    next_obs, reward, terminated, truncated, next_info = env.step(action.index())
    next_state = env.get_state()
    return next_state, reward, terminated

# Methods of detecting welfare reductions        
class WelfareMeasure:
    def __init__(self, discount_factor: float, discount_forward: bool, agent_policy: 'Policy' = None, env: 'EnvRenderWrapper' = None, save_estimated_trajectories: bool = False):
        self.discount_factor = discount_factor
        self.discount_forward = discount_forward
        self.agent_policy = agent_policy
        self.env = env
        self.save_estimated_trajectories = save_estimated_trajectories

        if self.save_estimated_trajectories:
            self.estimated_trajectories = {}
        
    def measure_from_trajectory(self, whole_trajectory: 'Trajectory', current_index: int) -> float:
        context_subtrajectory = self._get_context_subtrajectory(whole_trajectory, current_index)
        welfare = self._calculate_welfare(context_subtrajectory)
        return welfare
    
    def monte_carlo_estimate(self, whole_trajectory: 'Trajectory', current_index: int, num_rollouts: int, nao_support_policy: str, reduce: bool=True) -> float:
        # context_subtrajectory = self._get_context_subtrajectory(whole_trajectory, current_index)
        # context_start_state = context_subtrajectory[0].state
        # context_len = len(context_subtrajectory)

        if self.agent_policy is None and self.env is None:
            # This is for when there is no expected policy to compare reduced welfare
            return None
        
        if current_index is None:
            context_start_state = whole_trajectory[0].state
        else:
            context_start_state = whole_trajectory[current_index].state
        context_start_idx, context_end_idx = self._get_context_window(whole_trajectory, current_index)
        if context_end_idx == -1:
            max_len = None
        else:
            max_len = context_end_idx - context_start_idx

        welfare_estimates = []
        for i in range(num_rollouts):
            context_trajectory = rollout_trajectory(
                env=self.env,
                agent_policy=self.agent_policy,
                nao_support_policy=nao_support_policy,
                start_state=context_start_state,
                max_len=max_len,
            )
            if max_len is not None:
                assert len(context_trajectory) == max_len
            assert context_trajectory[0].state.equal(context_start_state)
            assert context_trajectory[0].state.time_state.frame == whole_trajectory[context_start_idx].state.time_state.frame
            assert context_trajectory[-1].state.time_state.frame == whole_trajectory[context_end_idx-1].state.time_state.frame
            #context_trajectory2 = rollout_trajectory(env=self.env,agent_policy=self.agent_policy,start_state=context_start_state,max_len=max_len-1,)

            if self.save_estimated_trajectories:
                key = f"current_step_index={current_index}__rollout_index={i}"
                self.estimated_trajectories[key] = context_trajectory

            welfare = self._calculate_welfare(context_trajectory)
            welfare_estimates.append(welfare)
        
        if reduce:
            return sum(welfare_estimates) / len(welfare_estimates)
        else:
            return welfare_estimates
        
    def _get_context_subtrajectory(self, whole_trajectory: 'Trajectory', current_index: int) -> 'Trajectory':
        start_idx, end_idx = self._get_context_window(whole_trajectory, current_index)
        context_subtrajectory = Trajectory(whole_trajectory[start_idx:end_idx])
        return context_subtrajectory
    
    def _get_discounted_sum_of_rewards(self, trajectory: 'Trajectory') -> float:
        total_reward = 0
        if self.discount_forward:
            for i, step in enumerate(trajectory):
                total_reward += step.reward * (self.discount_factor ** i)
        else:
            for i, step in enumerate(reversed(trajectory)):
                total_reward += step.reward * (self.discount_factor ** i)
        return total_reward
    
    def _calculate_welfare(self, context_trajectory: 'Trajectory') -> float:
        cumulative_discounted_reward = self._get_discounted_sum_of_rewards(context_trajectory)
        return cumulative_discounted_reward

    def _get_context_window(self, whole_trajectory: 'Trajectory', current_index: int) -> Tuple[int, int]:
        raise NotImplementedError("This method should be overridden by subclasses")


class CurrentRewardMeasure(WelfareMeasure):
    
    def __init__(self, discount_factor: float, short_term_horizon: int, agent_policy: 'Policy'=None, env: 'EnvRenderWrapper'=None):
        super().__init__(
            discount_factor=discount_factor,
            discount_forward=True,
            agent_policy=agent_policy,
            env=env
        )
        self.short_term_horizon = short_term_horizon 

    def _get_context_window(self, whole_trajectory: 'Trajectory', current_index: int) -> Tuple[int, int]:
        start_index = current_index
        end_index = min(len(whole_trajectory), start_index + self.short_term_horizon)
        return start_index, end_index
    
    # def _get_context_subtrajectory(self, whole_trajectory: 'Trajectory', current_index: int) -> 'Trajectory':
    #     start_index = current_index
    #     end_index = min(len(whole_trajectory), start_index + self.num_steps_in_horizon)
    #     context_subtrajectory = Trajectory(whole_trajectory[start_index:end_index])
    #     return context_subtrajectory
    
    # def _calculate_welfare_dev(self, context_trajectory: 'Trajectory') -> float:
    #     total_reward = 0
    #     #state, action, reward, info = traj_step.state, traj_step.action, traj_step.reward, traj_step.info
    #     for i, step in enumerate(context_trajectory):
    #         total_reward += step.reward * (self.discount_factor ** i)
        
    #     total_reward2 = self._get_discounted_sum_of_rewards(context_trajectory)
    #     assert total_reward == total_reward2
    #     return total_reward
    
class PastRewardsMeasure(WelfareMeasure):

    def __init__(self, discount_factor: float, short_term_horizon: int, agent_policy: 'Policy'=None, env: 'EnvRenderWrapper'=None):
        #super().__init__(discount_factor, agent_policy, env)
        super().__init__(
            discount_factor=discount_factor,
            discount_forward=False,
            agent_policy=agent_policy,
            env=env
        )
        self.short_term_horizon = short_term_horizon
    
    def _get_context_window(self, whole_trajectory: 'Trajectory', current_index: int) -> Tuple[int, int]:
        start_index = max(0, current_index - self.short_term_horizon)
        end_index = current_index
        return start_index, end_index
    

class FutureValueMeasure(WelfareMeasure):

    def __init__(self, discount_factor: float, short_horizon_before_valuefn: int, agent_policy: 'Policy', env: 'EnvRenderWrapper', value_function: ValueFunction):
        #super().__init__(discount_factor, agent_policy, env)
        super().__init__(
            discount_factor=discount_factor,
            discount_forward=True,
            agent_policy=agent_policy,
            env=env
        )
        self.short_horizon_before_valuefn = short_horizon_before_valuefn
        self.value_function = value_function

    def _get_context_window(self, whole_trajectory: 'Trajectory', current_index: int) -> Tuple[int, int]:
        start_index = current_index
        end_index = min(len(whole_trajectory), start_index + self.short_horizon_before_valuefn)
        return start_index, end_index
    
    def _calculate_welfare(self, context_trajectory: 'Trajectory') -> float:
        #return super()._calculate_welfare(context_trajectory)
        discounted_reward_over_context = self._get_discounted_sum_of_rewards(context_trajectory)
        discount_after_context = self.discount_factor ** len(context_trajectory)
        context_ending_states = context_trajectory.get_recent_states(index=-1, num_recent_states=ObservationConsts.RECENT_STATES_BUFFER_SIZE)
        future_value = self.value_function(context_ending_states)
        welfare = discounted_reward_over_context + discount_after_context * future_value
        return welfare

### NEW UPDATED WELFARE MEASURES

class NonLocalizedComponentRewardMeasure(WelfareMeasure):
    def __init__(self, agent_policy: 'Policy', env: 'EnvRenderWrapper', set_winner_at_trajectory_end: bool, save_estimated_trajectories: bool = False):
        super().__init__(
            discount_factor=1,
            discount_forward=True, # irrelevant if discount factor is 1
            agent_policy=agent_policy,
            env=env,
            save_estimated_trajectories=save_estimated_trajectories
        )
        self.set_winner_at_trajectory_end = set_winner_at_trajectory_end

    def _get_context_window(self, whole_trajectory: 'Trajectory', current_index: int) -> Tuple[int, int]:
        if current_index is not None:
            raise ValueError("NonLocalizedRewardMeasure does not have a current index")
        return 0, len(whole_trajectory)

    def _calculate_welfare(self, context_trajectory: 'Trajectory') -> WelfareInfo:
        if self.set_winner_at_trajectory_end:
            context_trajectory.set_winner_at_trajectory_end()

        all_welfare_step_info = []
        for step in context_trajectory:
            all_welfare_step_info.append(WelfareStepInfo.from_step_info(step.info))
        
        welfare_info = WelfareInfo.from_all_steps(
            all_welfare_step_info=all_welfare_step_info,
            initial_state=context_trajectory[0].state,
            final_state=context_trajectory[-1].state,
        )
        return welfare_info


class NonLocalizedRewardMeasure(WelfareMeasure):
    # there is no current index; assert that it must be None. Evaluate cumulative discounted rewards over entire trajectory
    # require that discount factor must be 1 (aka no discounting)

    def __init__(self, discount_factor: float, agent_policy: 'Policy', env: 'EnvRenderWrapper', set_winner_at_trajectory_end: bool):
        super().__init__(
            discount_factor=discount_factor,
            discount_forward=True, # this is irrelevant
            agent_policy=agent_policy,
            env=env
        )
        #raise NotImplementedError("This measure has not been tested yet. Remove this exception after testing")
        if discount_factor != 1:
            raise ValueError("NonLocalizedRewardMeasure requires discount factor to be 1")
        self.set_winner_at_trajectory_end = set_winner_at_trajectory_end
        
    def _get_context_window(self, whole_trajectory: 'Trajectory', current_index: int) -> Tuple[int, int]:
        if current_index is not None:
            raise ValueError("NonLocalizedRewardMeasure does not have a current index")
        # start_index = whole_trajectory[0].state.time_state.frame
        # end_index = whole_trajectory[-1].state.time_state.frame + 1 # +1 because end_index is exclusive
        # return start_index, end_index
        
        #return 0, -1
        return 0, len(whole_trajectory)
    
    def _calculate_welfare(self, context_trajectory) -> float:
        if self.set_winner_at_trajectory_end:
            context_trajectory.set_winner_at_trajectory_end()
        return super()._calculate_welfare(context_trajectory)
    
class CompletePastRewardsMeasure(WelfareMeasure):
    # optionally discounted rewards over the entire past trajectory, without a horizon size

    def __init__(self, discount_factor: float, agent_policy: 'Policy', env: 'EnvRenderWrapper', num_cached_rollouts: int = None, start_state: 'SpaceInvadersState' = None):
        super().__init__(
            discount_factor=discount_factor,
            discount_forward=False,
            agent_policy=agent_policy,
            env=env
        )
        if num_cached_rollouts is not None:
            self.monte_carlo_estimate = self._cached_monte_carlo_estimate

            rollout_trajectories = []
            for i in range(num_cached_rollouts):
                trajectory = rollout_trajectory(
                    env=self.env,
                    agent_policy=self.agent_policy,
                    start_state=start_state,
                    max_len=None,
                )
                rollout_trajectories.append(trajectory)
            self.cached_trajectories = rollout_trajectories
        #raise NotImplementedError("This measure has not been tested yet. Remove this exception after testing")

    def _get_context_window(self, whole_trajectory: 'Trajectory', current_index: int) -> Tuple[int, int]:
        # start_index = whole_trajectory[0].state.time_state.frame
        # end_index = current_index
        
        start_index = 0
        end_index = current_index
        return start_index, end_index
    
    def _cached_monte_carlo_estimate(self, whole_trajectory: 'Trajectory', current_index: int, num_rollouts: int, nao_support_policy: str=None, reduce: bool=True) -> float:
        # context_subtrajectory = self._get_context_subtrajectory(whole_trajectory, current_index)
        # context_start_state = context_subtrajectory[0].state
        # context_len = len(context_subtrajectory)

        if self.agent_policy is None and self.env is None:
            # This is for when there is no expected policy to compare reduced welfare
            return None
        
        # context_start_state = whole_trajectory[current_index].state
        # context_start_idx, context_end_idx = self._get_context_window(whole_trajectory, current_index)
        
        welfare_estimates = []
        for i in range(num_rollouts):
            #context_trajectory = self._get_context_window(self.cached_trajectories[i], current_index)
            
            # context_trajectory = self.cached_trajectories[i][context_start_idx:context_end_idx]
            # welfare = self._calculate_welfare(context_trajectory)
            # welfare_estimates.append(welfare)

            welfare = self.measure_from_trajectory(
                whole_trajectory=self.cached_trajectories[i],
                current_index=current_index,
            )
            welfare_estimates.append(welfare)

        if reduce:
            return sum(welfare_estimates) / len(welfare_estimates)
        else:
            return welfare_estimates

class SingleStepRewardMeasure(WelfareMeasure):
    def __init__(self, agent_policy: 'Policy', env: 'EnvRenderWrapper'):
        super().__init__(
            discount_factor=1, # irrelevant
            discount_forward=False, # irrelevant
            agent_policy=agent_policy,
            env=env
        )
        # not yet tested
        #raise NotImplementedError("This measure has not been tested yet. Remove this exception after testing")

    def _get_context_window(self, whole_trajectory: 'Trajectory', current_index: int) -> Tuple[int, int]:
        start_index = current_index
        end_index = current_index + 1
        return start_index, end_index
    
# TODO: resolve questions about future value measure
    # version 1: 
        # actual = take short horizon of actual trajectory, then calculate future value based on expected policies
        # expected = take short horizon of expected trajectory, then calculate future value based on expected policies
    # version 2:
        # actual = calculate future value based on actual policy for the whole trajectory
        # expected = calculate future value based on expected policy for the whole trajectory
# class FutureValueMeasure(WelfareMeasure):

#     def __init__(self, discount_factor: float, agent_policy: 'Policy', env: 'EnvRenderWrapper', value_function: ValueFunction):
#         super().__init__(discount_factor, agent_policy, env)
#         self.value_function = value_function

#     def _get_context_window(self, whole_trajectory: 'Trajectory', current_index: int) -> Tuple[int, int]:
#         start_index = current_index
#         end_index = -1
#         return start_index, end_index
    
#     def _calculate_welfare(self, trajectory: 'Trajectory') -> float:
#         total_reward = 0
#         for i, step in enumerate(trajectory):
#             total_reward += step.reward * (self.discount_factor ** i)
#         total_reward2 = self._get_discounted_sum_of_rewards(trajectory)
#         assert total_reward == total_reward2
#         assert i+1 == len(trajectory)

#         recent_states = trajectory.get_recent_states(index=-1, num_recent_states=ObservationConsts.RECENT_STATES_BUFFER_SIZE)
#         last_state_value = self.value_function(recent_states)
#         current_state_value = total_reward + (self.discount_factor**(i+1) * last_state_value)
#         return current_state_value

TESTING_NEW_MEASURE_CODE = False
def current_reward(state: 'SpaceInvadersState', agent_policy: 'Policy', env: 'EnvRenderWrapper', short_term_horizon: int, discount_factor: float) -> float:
    raise NotImplementedError("This function is deprecated")
    
    _start_state = state

    if TESTING_NEW_MEASURE_CODE:
        from utils import set_seed, set_deterministic_cudnn
        set_seed(seed=42)
        set_deterministic_cudnn()
    
    total_reward = 0
    for i in range(short_term_horizon):
        state, reward, terminated = step_from_state(env, state, agent_policy(state))
        total_reward += reward * (discount_factor ** i)
        if terminated:
            break


    if TESTING_NEW_MEASURE_CODE:
        set_seed(seed=42)
        set_deterministic_cudnn()
        total_reward2 = 0
        trajectory = Trajectory()
        for i in range(short_term_horizon):
            action = agent_policy(state) 
            next_state, reward, terminated = step_from_state(env, state, action)
            trajectory.add_step(state, action, reward, {"step_info": {}})
            total_reward2 += reward * (discount_factor ** i)
            state = next_state
            if terminated:
                break
        
        measure = CurrentRewardMeasure(discount_factor, short_term_horizon, agent_policy, env)
        total_reward3 = measure._calculate_welfare(trajectory)
        assert total_reward2 == total_reward3
        #assert total_reward == total_reward2
        #import pdb; pdb.set_trace()
    
    return total_reward

def future_value(state: 'SpaceInvadersState', agent_policy: 'Policy', env: 'EnvRenderWrapper', value_function: ValueFunction, short_term_horizon: int) -> float:    
    raise NotImplementedError("This function is deprecated")
    
    total_reward = 0
    discount_factor = value_function.discount_factor
    trajectory = Trajectory()
    for i in range(short_term_horizon):
        action = agent_policy(state)
        next_state, reward, terminated = step_from_state(env, state, action)
        trajectory.add_step(state, action, reward, {"step_info": {}})
        total_reward += reward * (discount_factor ** i)
        state = next_state
        if terminated:
            break
    
    recent_states = trajectory.get_recent_states(index=-1, num_recent_states=ObservationConsts.RECENT_STATES_BUFFER_SIZE)
    last_state_value = value_function(recent_states)
    current_state_value = total_reward + (discount_factor**(i+1) * last_state_value)
    
    return current_state_value

def sum_past_rewards(past_state: 'SpaceInvadersState', agent_policy: 'Policy', env: 'EnvRenderWrapper', discount_factor: float, num_steps: float) -> float:
    raise NotImplementedError("This function is deprecated")
    
    total_reward = 0
    state = past_state
    for i in range(num_steps):
        next_state, reward, terminated = step_from_state(env, state, agent_policy(state))
        total_reward += discount_factor**i * reward
        state = next_state
        if terminated:
            break
    return total_reward

def sum_past_rewards_and_future_value(past_state: 'SpaceInvadersState', agent_policy: 'Policy', env: 'EnvRenderWrapper', num_steps: float, value_function: ValueFunction) -> float:
    raise NotImplementedError("This function is deprecated")
    
    total_reward = 0
    discount_factor = value_function.discount_factor
    trajectory = Trajectory()
    state = past_state
    for i in range(num_steps):
        action = agent_policy(state)
        next_state, reward, terminated = step_from_state(env, state, action)
        trajectory.add_step(state, action, reward, {"step_info": {}})
        total_reward += discount_factor**i * reward
        state = next_state
        if terminated:
            break

    recent_states = trajectory.get_recent_states(index=-1, num_recent_states=ObservationConsts.RECENT_STATES_BUFFER_SIZE)
    total_reward += discount_factor**(i+1) * value_function(recent_states)
    return total_reward

@dataclass
class TrajectoryStep:
    state: 'SpaceInvadersState'
    #recent_states: Tuple['SpaceInvadersState', ...]
    action: 'Action'
    reward: float
    info: 'StepInfo'

# TODO? update Action to include all agents' actions
class Trajectory:
    #def __init__(self, num_stacked_steps: int = 1):
    def __init__(self, steps: List[TrajectoryStep] = None, config: 'SpaceInvadersConfig' = None):
        if steps is not None:
            self.steps = steps
        else:
            self.steps: List[TrajectoryStep] = []

        self.config = config
        
        # from collections import deque
        # if num_stacked_steps > 1:
        #     self.recent_states = deque(maxlen=num_stacked_steps)
        # else:
        #     self.recent_states = None

    def add_step(self, state, action, reward, info):
        # if self.recent_states is not None:
        #     self.recent_states.append(state)
        # import copy
        # recent_states = tuple(copy.deepcopy(self.recent_states)) if self.recent_states is not None else None
        
        step = TrajectoryStep(
            state=state, 
            #recent_states=recent_states,
            action=action,
            reward=reward,
            info=info["step_info"]
        )
        self.steps.append(step)

    def is_terminal(self) -> bool:
        if len(self.steps) == 0:
            return False
        final_state: SpaceInvadersState = self.steps[-1].state

        # TODO? add any other parsing checks?

        return final_state.is_terminal(self.config)
    
    def set_winner_at_trajectory_end(self):
        final_state = self.steps[-1].state
        final_step_info = self.steps[-1].info
        
        # If final state of trajectory is already a game-terminal state, do nothing and return
        if final_state.is_terminal(self.config):
            return

        winning_player = final_state.get_leading_score_player(threshold=0)
        if winning_player not in [Players.HUMAN, Players.SHUTTER]:
            raise ValueError(f"Winning player {winning_player} is not a valid player")
        
        # # Check that player victory rewards is zero for all players
        # for player in Players:
        #     if final_step_info.player_victory_rewards[player] != 0:
        #         raise ValueError(f"Player {player} has non-zero victory reward at the end of trajectory")
        
        # final_step_info.player_victory_rewards[winning_player] += RewardConsts.VICTORY_REWARD

        for player in Players:
            # Note: if winner is already set at final state, then this will keep the same reward
            if player == winning_player:
                final_step_info.player_victory_rewards[player] = RewardConsts.VICTORY_REWARD
            else:
                if final_step_info.player_victory_rewards[player] != 0:
                    raise ValueError(f"Player {player} has non-zero victory reward at the end of trajectory")
    
    def get_recent_states(self, index: int, num_recent_states: int) -> Tuple['SpaceInvadersState', ...]:
        return tuple(self.steps[index - num_recent_states + i].state for i in range(num_recent_states))   
 
    def __iter__(self) -> Iterator[TrajectoryStep]:
        return iter(self.steps)
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def __getitem__(self, index: int) -> TrajectoryStep:
        return self.steps[index]

    def __eq__(self, value: object) -> bool:
        try:
            return self.equal(value, verbose=False)
        except:
            return False
    
    def equal(self, value: 'Trajectory', verbose: bool = False) -> bool:
        if not isinstance(value, Trajectory):
            raise NotImplementedError("Only comparison with Trajectory is supported")
        #return self.steps == value.steps

        if len(self.steps) != len(value.steps):
            if verbose:
                print("Lengths are different")
            return False
        
        for i, (step1, step2) in enumerate(zip(self.steps, value.steps)):
            if step1.state.equal(step2.state, verbose=verbose) == False:
                if verbose:
                    print(f"States are different at index {i}")
                return False
            if step1.action != step2.action:
                if verbose:
                    print(f"Actions are different at index {i}")
                return False
            if step1.reward != step2.reward:
                if verbose:
                    print(f"Rewards are different at index {i}")
                return False
            if step1.info != step2.info:
                if verbose:
                    print(f"Infos are different at index {i}")
                return False

        return True
        

    # NOTE: use this function to fill in any missing values such as state variables added after trajectory recording
    def _fill_missing_values(self):

        FILL_NAO_SUPPORTING_PLAYER = False
        if FILL_NAO_SUPPORTING_PLAYER:
            for i, step in enumerate(self.steps):
                if i > 0:
                    prev_state = self.steps[i-1].state
                    state = step.state
                    if state.players_state.nao_supporting_player is None:
                        state.players_state.nao_supporting_player = prev_state.players_state.nao_supporting_player


    def to_json(self) -> dict:
        dct = {
            # States
            "states": [step.state.to_dict() for step in self.steps],
            # Actions
            "actions": [step.action for step in self.steps],
            # Rewards
            "rewards": [step.reward for step in self.steps],
            # Info 
            "info": [step.info.to_dict() for step in self.steps]
        }

        return {
            'metadata': {
                'file_structure': 'v1',
                'config': self.config.to_dict() if self.config is not None else None,},
            **dct
        }

    @classmethod
    def from_json(cls, traj_json: dict) -> 'Trajectory':
        if 'metadata' in traj_json:
            file_structure = traj_json['metadata']['file_structure']
            config_json = traj_json['metadata'].get('config', None)
        else:
            file_structure = 'legacy'
            config_json = None
        
        if config_json is not None:
            config = SpaceInvadersConfig.from_dict(config_json)
        traj = Trajectory(config=config)
        for state_json, action_json, reward, info_json in zip(traj_json['states'], traj_json['actions'], traj_json['rewards'], traj_json['info']):
            if file_structure == 'legacy':
                state_json['bullet_state']['enemy_bullets'] = np.asarray(state_json['bullet_state']['enemy_bullets'])
                state_json['bullet_state']['player_bullets'] = np.asarray(state_json['bullet_state']['player_bullets'])
                state_json['bullet_state']['bullet_y_positions'] = np.asarray(state_json['bullet_state']['bullet_y_positions'])
                recursive_remove_key(state_json, '_init_done')
                recursive_remove_key(state_json, '_field_names')

                info_json['human_actions_taken'] = SingleAgentAction(**info_json['human_actions_taken'])
                info_json['shutter_actions_taken'] = SingleAgentAction(**info_json['shutter_actions_taken'])
                info_json['nao_actions_taken'] = SingleAgentAction(**info_json['nao_actions_taken'])

                state = SpaceInvadersState.from_dict(state_json, legacy_file_structure=True)
                action = SingleAgentAction.from_dict(action_json)
                step_info = StepInfo(); step_info.from_dict(info_json) # TODO: make this a classmethod that returns an instance
                traj.add_step(state=state, action=action, reward=reward, info={"step_info": step_info})
            elif file_structure == 'v1':
                state = SpaceInvadersState.from_dict(state_json)
                action = action_json
                step_info = StepInfo.from_dict(info_json)
                traj.add_step(state=state, action=action, reward=reward, info={"step_info": step_info})

        return traj

    @classmethod
    def from_file(cls, traj_file: str, fill_missing_values: bool = False) -> 'Trajectory':
        with open(traj_file, 'r') as file:
            traj_json = json.load(file, object_hook=CustomJSONDecoder.custom_json_decode)
        
        trajectory = cls.from_json(traj_json)
        if fill_missing_values:
            trajectory._fill_missing_values()
        return trajectory
    
    def to_file(self, traj_file: str):
        with open(traj_file, 'w') as file:
            json.dump(self.to_json(), file, cls=CustomJSONEncoder)
        return traj_file

def mc_welfare2(welfare_function: Callable, num_rollouts: int, reduce=True) -> float:
    raise NotImplementedError("This function is deprecated")
    
    welfare_estimates = [welfare_function() for _ in range(num_rollouts)]
    if reduce:
        return sum(welfare_estimates) / len(welfare_estimates)
    else:
        return welfare_estimates


def rollout_interactive_trajectory(env: 'EnvRenderWrapper', start_state:SpaceInvadersState=None, max_len:int=None,
                                   debugging_skip_pauses: bool = False,
                                   num_game_segments:int=DynamicsConsts.NUM_GAME_SEGMENTS, game_duration_frames: int=DynamicsConsts.MAX_NUM_FRAMES_PER_GAME,
                                   #interactive_players:List[Players]=[Players.HUMAN, Players.NAO],
                                   robot_action_queue: 'RobotActionQueue'=None) -> 'Trajectory':
    """
    Rollout a trajectory interactively using keyboard inputs for two players.

    Args:
        env (EnvRenderWrapper): The environment to interact with.
        agent_policy (Policy): The policy for the agent (must be SingleAgentPolicy).
        start_state (dict, optional): The initial state of the environment. Defaults to None.
        max_len (int, optional): The maximum length of the trajectory. Defaults to None.

    Returns:
        Trajectory: The trajectory of the interactive gameplay.
    """

    raise NotImplementedError("This function is deprecated; use play_interactive_game.py instead")

    interactive_mode: InteractiveModes = env.unwrapped.config.interactive_mode
    assert interactive_mode != InteractiveModes.NONE, "Interactive mode must be specified"
    
    action_space = env.unwrapped.config.action_space
    if action_space != ActionSpaces.HUMAN_ONLY:
        raise NotImplementedError("Action space must be HUMAN_ONLY for interactive mode, until properly converting to joint action space.")

    if debugging_skip_pauses:
        dialog_kwargs = {"forced_wait_time_seconds": 0}
    else:
        dialog_kwargs = {}
    
    RANDOM_TIEBREAK = True
    trajectory = Trajectory()
    terminated = False
    
    # Initialize the environment
    obs, info = env.reset()
    if start_state is not None:
        env.set_state(start_state)
    
    # Initialize player actions and define key mappings
    if interactive_mode == InteractiveModes.KEYBOARD:
        left_keyboard = PyGameKeyboard(
            mapping_type=PyGameKeyboard.MappingTypes.LEFT_PLAYER,
        )
        right_keyboard = PyGameKeyboard(
            mapping_type=PyGameKeyboard.MappingTypes.RIGHT_PLAYER,
        )
    elif interactive_mode == InteractiveModes.GAMEPAD:
        
        TESTING_WITH_ONE_GAMEPAD = True
        # If testing with one gamepad, use the same gamepad for both players
        # If testing with two gamepads, use one for each player

        if TESTING_WITH_ONE_GAMEPAD:
            left_gamepad = PyGameGamepad(
                gamepad_index=0, mapping_type=PyGameGamepad.MappingTypes.SPLIT_GAMEPAD_LEFT_PLAYER
            )
            right_gamepad = PyGameGamepad(
                gamepad_index=0, mapping_type=PyGameGamepad.MappingTypes.SPLIT_GAMEPAD_RIGHT_PLAYER
            )
        else:
            left_gamepad = PyGameGamepad(
                gamepad_index=0, mapping_type=PyGameGamepad.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER
            )
            right_gamepad = PyGameGamepad(
                gamepad_index=1, mapping_type=PyGameGamepad.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER
            )

    else:
        raise NotImplementedError("Interactive mode must be KEYBOARD or GAMEPAD")

    if num_game_segments == 1:
        POWERUPS = None
    elif num_game_segments == 3:
        #NUM_FRAMES_PER_GAME = DynamicsConsts.MAX_NUM_FRAMES_PER_GAME
        NUM_FRAMES_PER_SEGMENT = int(game_duration_frames // num_game_segments)
        
        # POWERUPS = random.sample(["left_player_double_bullet", "right_player_double_bullet"], 2)
        # POWERUPS.append("none")  # Add "none" powerup for the last segment
        # print(f"Powerups for segments: {POWERUPS}")

        POWERUPS = ["left_player_double_bullet", "right_player_double_bullet", "none"]
    else:
        raise NotImplementedError("Number of game segments must be 1 or 3 for interactive mode")
    
    # TODO: move blocking message popups into the competitive env class
    while not terminated:
        # Enforce frame rate
        env.unwrapped.clock.tick(DynamicsConsts.FRAMES_PER_SECOND)

        # Get the current state
        state: SpaceInvadersState = env.get_state()

        # Check if entering a new game segment
        if num_game_segments > 1 and state.time_state.frame % NUM_FRAMES_PER_SEGMENT == 0:
            current_segment = state.time_state.frame // NUM_FRAMES_PER_SEGMENT
            print(f"Current segment: {current_segment} (frame {state.time_state.frame})")

            current_powerup = POWERUPS[current_segment]
            env.unwrapped.set_boost(current_powerup)

            if state.time_state.frame == 0:
                # Beginning of the first segment
                # env.unwrapped.display_blocking_dialog_message(
                #     message="Welcome to the game!\n\nPress any key to start.",
                #     interactive_mode=interactive_mode,
                # )
                message1 = "Welcome to the game!\n\nPress any button to continue."

                if current_powerup == "left_player_double_bullet":
                    message2 = f"\n\nFor this first segment, the player on the left gets a random double bullet boost!"
                elif current_powerup == "right_player_double_bullet":
                    message2 = f"\n\nFor this first segment, the player on the right gets a random double bullet boost!"
                else:
                    raise NotImplementedError("Powerup not implemented yet")
                message2 += "\n\nPress any button to start the game."
            elif current_segment <= num_game_segments:
                # Boundary for a new segment
                # env.unwrapped.display_blocking_dialog_message(
                #     message=(
                #         f"Game paused after segment {current_segment - 1} of {NUM_GAME_SEGMENTS}.\n"
                #         "Please answer the survey questions.\n"
                #         "When you're ready, press any key to continue."
                #     ),
                #     interactive_mode=interactive_mode,
                # )
                message1 = (
                    f"Game paused after segment {current_segment} of {num_game_segments}.\n"
                    "Please answer the survey questions.\n"
                    "When you're ready, press any button to continue."
                )

                if current_powerup == "left_player_double_bullet":
                    message2 = f"\n\nFor this next segment, the player on the left gets a random double bullet boost!"
                elif current_powerup == "right_player_double_bullet":
                    message2 = f"\n\nFor this next segment, the player on the right gets a random double bullet boost!"
                elif current_powerup == "none":
                    message2 = f"\n\nFor this next segment, no player gets a random double bullet boost!"
                else:
                    raise NotImplementedError("Powerup not implemented yet")
                message2 += "\n\nPress any button to resume the game."
            else:
                raise ValueError("Current segment is greater than the number of game segments")
            
            # robot faces the players
            robot_action_queue.add_robot_action(RobotActionRequest(
                type=RobotActionType.MOVEMENT,
                params={"position_name": "resting"},
            ))
            # robot speaks
            if current_segment > 0:
                # robot speaks
                robot_action_queue.add_robot_action(RobotActionRequest(
                    type=RobotActionType.SPEECH,
                    params={"message": "Let's take a break and answer some questions."},
                ))
            # game displays blocking message 1
            env.unwrapped.display_blocking_dialog_message(
                message=message1,
                interactive_mode=interactive_mode,
                countdown_seconds=0,
                **dialog_kwargs
            )

            if message2 is not None:
                # robot faces the game screen
                robot_action_queue.add_robot_action(RobotActionRequest(
                    type=RobotActionType.MOVEMENT,
                    params={"position_name": "look_at_screen_without_blocking"},
                ))
                if current_segment > 0:
                    # robot speaks
                    robot_action_queue.add_robot_action(RobotActionRequest(
                        type=RobotActionType.SPEECH,
                        params={"message": "Let's get back to the game!"},
                    ))
                # game displays blocking message 2
                env.unwrapped.display_blocking_dialog_message(
                    message=message2,
                    interactive_mode=interactive_mode,
                    countdown_seconds=5 if not debugging_skip_pauses else 0,
                    **dialog_kwargs
                )



        # Get actions for each player from keyboard inputs
        events = pygame.event.get()
        if interactive_mode == InteractiveModes.KEYBOARD:
            left_player_action = left_keyboard.parse_action(events)
            right_player_action = right_keyboard.parse_action(events)
        else:
            # For gamepad inputs, parse the actions from the gamepad
            left_player_action = left_gamepad.parse_action(events)
            right_player_action = right_gamepad.parse_action(events)

        joint_action = {
            'left_player_action_index': left_player_action.tie_breaker(inplace=False, random_tiebreak=RANDOM_TIEBREAK).index(),
            'right_player_action_index': right_player_action.tie_breaker(inplace=False, random_tiebreak=RANDOM_TIEBREAK).index(),
        }

        # Step the environment
        # NOTE: trajectory records action as one player arbitrarily, but info has actions of every agent
        # TODO: properly integrate joint action of all agents into environment step function and trajectory recording
        next_obs, reward, terminated, truncated, info = env.step(joint_action)
        trajectory.add_step(state, left_player_action, reward, info)

        step_info: 'StepInfo' = info["step_info"]
        if robot_action_queue is not None:
            if step_info.requested_robot_action is not None:
                # Add the requested robot action to the queue
                # This is used for the physical robot to take actions outside of the game environment
                robot_action_queue.add_robot_action(
                    step_info.requested_robot_action
                )
        if interactive_mode == InteractiveModes.GAMEPAD:
            if Players.HUMAN in step_info.players_hit:
                left_gamepad.rumble()
            if Players.SHUTTER in step_info.players_hit:
                right_gamepad.rumble()

            

        # Optional early termination
        if max_len is not None:
            if max_len < 0:
                raise ValueError("max_len must be a non-negative integer. Use None for no limit.")
            if len(trajectory) >= max_len:
                break
    
    # End of the last segment
    winning_player = state.get_leading_score_player(threshold=0)
    winning_player_description = "left" if winning_player == Players.HUMAN else "right"
    # env.unwrapped.display_blocking_dialog_message(
    #     message=(
    #         f"Game over after segment {num_game_segments} of {num_game_segments}.\n"
    #         "Thank you for playing!"
    #     ),
    #     interactive_mode=interactive_mode,
    # )
    # robot faces the players
    robot_action_queue.add_robot_action(RobotActionRequest(
        type=RobotActionType.MOVEMENT,
        params={"position_name": "resting"},
    ))
    message = (
        f"Game over after segment {num_game_segments} of {num_game_segments}.\n"
        f"The player on the {winning_player_description} won, congratulations!\n"
        "Thank you for playing!"
    )
    env.unwrapped.display_blocking_dialog_message(
        message=message,
        interactive_mode=interactive_mode,
        countdown_seconds=0,
        **dialog_kwargs
    )
    
    return trajectory


# Rollout the trajectory
def rollout_trajectory(env: 'EnvRenderWrapper', agent_policy: 'Policy', nao_support_policy: str=None, start_state=None, max_len=None, debug_break_after_reward=False) -> 'Trajectory':

    trajectory = Trajectory()
    terminated = False
    obs, info = env.reset()
    if nao_support_policy == None:
        env.reset_config_nao_policy()
    else:
        env.new_nao_policy(nao_support_policy)
    steps_until_break = None

    if start_state is not None:
        #env.state = env.unwrapped.state = start_state
        env.set_state(start_state)
    while not terminated:
        state = env.get_state()
        action = agent_policy(state)
        next_obs, reward, terminated, truncated, info = env.step(action.index())
        trajectory.add_step(state, action, reward, info)
        
        # Temporary shortcut for debugging
        if debug_break_after_reward:
            if reward != 0:
                #print("reward = ", reward)
                steps_until_break = 25
            if steps_until_break is not None:
                steps_until_break -= 1
                if steps_until_break == 0:
                    break

        # Temporary shortcut for debugging
        if max_len is not None:
            if max_len < 0:
                raise ValueError("max_len must be a non-negative integer. Use None for no limit.")
            if len(trajectory) >= max_len:
                break
    
    return trajectory

def get_discounted_future_rewards_at_every_step(trajectory: 'Trajectory', discount_factor: float) -> List[float]:
    discounted_future_rewards_by_step = [0] * len(trajectory)
    future_reward = 0

    for i in reversed(range(len(trajectory))):
        step = trajectory[i]
        future_reward = step.reward + discount_factor * future_reward
        discounted_future_rewards_by_step[i] = future_reward

    VERIFY = False
    if VERIFY:
        discounted_future_rewards_by_step_slow = get_discounted_future_rewards_at_every_step_slow(trajectory, discount_factor)
        # print(f"discounted_future_rewards_by_step = {discounted_future_rewards_by_step}")
        # print(f"discounted_future_rewards_by_step_slow = {discounted_future_rewards_by_step_slow}")
        # print(f"rewards = {[step.reward for step in trajectory]}")
        import numpy as np
        assert np.allclose(discounted_future_rewards_by_step, discounted_future_rewards_by_step_slow)

    return discounted_future_rewards_by_step


def get_discounted_future_rewards_at_every_step_slow(trajectory: 'Trajectory', discount_factor: float) -> List[float]:
    discounted_future_rewards_by_step = [0] * len(trajectory)
    for i in range(len(trajectory)):
        discounted_future_reward = 0
        current_discount_factor = 1
        for j in range(i, len(trajectory)):
            step = trajectory[j]
            #if step.reward > 0 and (j == i+1 or j == i) : import pdb; pdb.set_trace()
            discounted_future_reward += step.reward * current_discount_factor
            current_discount_factor *= discount_factor  # Increase the discount factor for the next step
        discounted_future_rewards_by_step[i] = discounted_future_reward

    return discounted_future_rewards_by_step

TEMP_IGNORE_VALUE_FUNCTION = False

def save_welfare_values(output_dir, filename, nonlocalized_reward_values, single_step_reward_values, complete_past_rewards_values, nonlocalized_component_reward_values, metadata=None):
    # serializable_nonlocalized_component_reward_values = [
    #     (frame, actual.to_json(), expected.to_json()) for frame, actual, expected in nonlocalized_component_reward_values
    # ]
    # #frame, actual_nonlocalized_component_reward, expected_nonlocalized_component_reward
    
    # data = {
    #     "metadata": metadata if metadata else {},
    #     "welfare_measures": {
    #         "nonlocalized_reward_values": nonlocalized_reward_values,
    #         "single_step_reward_values": single_step_reward_values,
    #         "complete_past_rewards_values": complete_past_rewards_values,
    #         "nonlocalized_component_reward_values": serializable_nonlocalized_component_reward_values,
    #     }
    # }
    # with open(os.path.join(output_dir, filename), 'w') as f:
    #     json.dump(data, f)


    # # test loading
    # with open(os.path.join(output_dir, filename), 'r') as f:
    #     data_loaded = json.load(f)
    # loaded_component_values = [
    #     (frame, WelfareInfo.from_json(actual_json), WelfareInfo.from_json(expected_json))
    #     for frame, actual_json, expected_json in data_loaded["welfare_measures"]["nonlocalized_component_reward_values"]
    # ]
    # import pdb; pdb.set_trace()


    serializable_nonlocalized_component_reward_values = [
        (
            frame,
            actual.to_json(),
            [expected.to_json() for expected in multiple_expected]
        ) for frame, actual, multiple_expected in nonlocalized_component_reward_values
    ]
    #frame, actual_nonlocalized_component_reward, expected_nonlocalized_component_reward
    
    data = {
        "metadata": metadata if metadata else {},
        "welfare_measures": {
            "nonlocalized_reward_values": nonlocalized_reward_values,
            "single_step_reward_values": single_step_reward_values,
            "complete_past_rewards_values": complete_past_rewards_values,
            "nonlocalized_component_reward_values": serializable_nonlocalized_component_reward_values,
        }
    }
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(data, f)


    # test loading
    # with open(os.path.join(output_dir, filename), 'r') as f:
    #     data_loaded = json.load(f)
    # loaded_component_values = [
    #     (
    #         frame,
    #         WelfareInfo.from_json(actual_json),
    #         [WelfareInfo.from_json(expected_json) for expected_json in multiple_expected_json]
    #     )
    #     for frame, actual_json, multiple_expected_json in data_loaded["welfare_measures"]["nonlocalized_component_reward_values"]
    # ]
    # NOTE: every frame is the same because this reward measure is non-localized, aka defined over the entire trajectory
        # Take an arbitrary frame to use the welfare values

def detect_welfare_reduction2(
    trajectory: 'Trajectory',
    actual_agent_policy: 'Policy',
    expected_agent_policy: 'Policy',
    actual_env: 'EnvRenderWrapper',
    expected_env: 'EnvRenderWrapper',
    nao_support_policy: str,
    human_expectations_value_function: ValueFunction,
    retrospective_length: int,
    short_term_horizon: int,
    num_mc_rollouts: int,
    output_dir: str,
    welfare_plot_fn: callable,
    resume_from_output_dir: str = None,
    debugging_retrospective_cutoff: int = None,
    set_winner_at_trajectory_end: bool = False,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:

    # Using support policy that's embedded in actual/expected environments
    assert nao_support_policy == None
    
    WELFARE_VALUES_FILENAME = "welfare_values.json"

    if set_winner_at_trajectory_end:
        # based on which player has higher score at the end of the trajectory, set the winning reward
        trajectory.set_winner_at_trajectory_end()

    nonlocalized_reward_measure = NonLocalizedRewardMeasure(
        discount_factor=human_expectations_value_function.discount_factor,
        agent_policy=expected_agent_policy,
        env=expected_env,
        set_winner_at_trajectory_end=set_winner_at_trajectory_end,
    )
    actual_nonlocalized_reward = nonlocalized_reward_measure.measure_from_trajectory(
        whole_trajectory=trajectory,
        current_index=None,
    )
    expected_nonlocalized_reward = nonlocalized_reward_measure.monte_carlo_estimate(
        whole_trajectory=trajectory,
        current_index=None,
        num_rollouts=num_mc_rollouts,
        nao_support_policy=nao_support_policy
    )
    print(f"Nonlocalized reward measure done:\n\tActual: {actual_nonlocalized_reward}\n\tExpected: {expected_nonlocalized_reward}")

    single_step_reward_measure = SingleStepRewardMeasure(
        agent_policy=expected_agent_policy,
        env=expected_env,
    )
    
    complete_past_rewards_measure = CompletePastRewardsMeasure(
        discount_factor=human_expectations_value_function.discount_factor,
        agent_policy=expected_agent_policy,
        env=expected_env,
        num_cached_rollouts=num_mc_rollouts,
        start_state=trajectory[0].state,
    )

    nonlocalized_component_rewards_measure = NonLocalizedComponentRewardMeasure(
        agent_policy=expected_agent_policy,
        env=expected_env,
        set_winner_at_trajectory_end=set_winner_at_trajectory_end,
        save_estimated_trajectories=True,
    )
    actual_nonlocalized_component_reward = nonlocalized_component_rewards_measure.measure_from_trajectory(
        whole_trajectory=trajectory,
        current_index=None,
    )
    expected_nonlocalized_component_reward = nonlocalized_component_rewards_measure.monte_carlo_estimate(
        whole_trajectory=trajectory,
        current_index=None,
        num_rollouts=num_mc_rollouts,
        nao_support_policy=nao_support_policy,
        reduce=False
    )
    COMPONENT_REWARDS_TRAJ_BASENAME = "nonlocalized_component_rewards__estimated_trajectory"
    for k, traj in nonlocalized_component_rewards_measure.estimated_trajectories.items():
        traj_file = os.path.join(output_dir, f"{COMPONENT_REWARDS_TRAJ_BASENAME}__{k}.json")
        traj.to_file(traj_file)

    #all_welfare_values = []

    nonlocalized_reward_values = [(-1, actual_nonlocalized_reward, expected_nonlocalized_reward)]
    single_step_reward_values = []
    complete_past_rewards_values = []
    nonlocalized_component_rewards = [(-1, actual_nonlocalized_component_reward, expected_nonlocalized_component_reward)]
    for i, traj_step in enumerate(tqdm(trajectory, desc="Detecting welfare reduction over trajectory steps")):

        # if i % 5 != 0:
        #     continue

        #welfare_values = {}

        frame = traj_step.state.time_state.frame

        #nonlocalized_reward_values.append((frame, actual_nonlocalized_reward, expected_nonlocalized_reward))

        #nonlocalized_component_rewards.append((frame, actual_nonlocalized_component_reward, expected_nonlocalized_component_reward))
        
        actual_single_step_reward = single_step_reward_measure.measure_from_trajectory(
            whole_trajectory=trajectory,
            current_index=i,
        )
        expected_single_step_reward = single_step_reward_measure.monte_carlo_estimate(
            whole_trajectory=trajectory,
            current_index=i,
            num_rollouts=num_mc_rollouts,
            nao_support_policy=nao_support_policy
        )
        # if actual_single_step_reward != expected_single_step_reward:
        #     import pdb; pdb.set_trace()
        single_step_reward_values.append((frame, actual_single_step_reward, expected_single_step_reward))

        
        actual_past_reward = complete_past_rewards_measure.measure_from_trajectory(
            whole_trajectory=trajectory,
            current_index=i,
        )
        # faster way: we're simulating multiple rollouts from each state
            # future rollouts can check if the current state has already been used in a rollout, and steal that remaining rollout
        expected_past_reward = complete_past_rewards_measure.monte_carlo_estimate(
            whole_trajectory=trajectory,
            current_index=i,
            num_rollouts=num_mc_rollouts,
            nao_support_policy=nao_support_policy
        )
        complete_past_rewards_values.append((frame, actual_past_reward, expected_past_reward))

        # if actual_past_reward > 0:
        #     import pdb; pdb.set_trace()
        
        #all_welfare_values.append(welfare_values)

        if i > 0 and i % 1000 == 0:
            welfare_plot_fn(
                nonlocalized_values = nonlocalized_reward_values,
                single_step_values = single_step_reward_values,
                past_values = complete_past_rewards_values,
                output_dir=output_dir,
                compare_smoothing=True,
            )

        if i >0 and i % 1000 == 0:
            # Save the welfare values every 500 steps, as json
            save_welfare_values(
                output_dir=output_dir,
                filename=WELFARE_VALUES_FILENAME,
                nonlocalized_reward_values=nonlocalized_reward_values,
                single_step_reward_values=single_step_reward_values,
                complete_past_rewards_values=complete_past_rewards_values,
                nonlocalized_component_reward_values=nonlocalized_component_rewards,
                metadata={
                    # TODO: anything else to track outside or args.json saved with the experiment command
                }
            )

    # final plot
    welfare_plot_fn(
        nonlocalized_values = nonlocalized_reward_values,
        single_step_values = single_step_reward_values,
        past_values = complete_past_rewards_values,
        output_dir=output_dir,
        compare_smoothing=True,
    )
    # TODO add component rewards to plot

    # final values
    save_welfare_values(
        output_dir=output_dir,
        filename=WELFARE_VALUES_FILENAME,
        nonlocalized_reward_values=nonlocalized_reward_values,
        single_step_reward_values=single_step_reward_values,
        complete_past_rewards_values=complete_past_rewards_values,
        nonlocalized_component_reward_values=nonlocalized_component_rewards,
        metadata={
            # TODO: anything else to track outside or args.json saved with the experiment command
        }
    )
    
    return nonlocalized_reward_values, single_step_reward_values, complete_past_rewards_values



# Test the methods of detecting welfare reductions
# TODO: class for welfare reduction detection methods, returning actual and expected values. Shared code for tracking/plotting
def detect_welfare_reduction(
    trajectory: 'Trajectory', 
    actual_agent_policy: 'Policy', 
    expected_agent_policy: 'Policy', 
    actual_env: 'EnvRenderWrapper', 
    expected_env: 'EnvRenderWrapper', 
    human_expectations_value_function: ValueFunction, 
    retrospective_length: int, 
    short_term_horizon: int, 
    num_mc_rollouts: int, 
    output_dir: str,
    welfare_plot_fn: callable,
    resume_from_output_dir: str = None,
    debugging_retrospective_cutoff: int = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:

    raise NotImplementedError("This function is deprecated")
    
    WELFARE_VALUES_FILENAME = "welfare_values.json"

    reward_values = []
    future_value_values = []
    past_rewards_values = []
    past_rewards_and_future_value_values = []
    if TEMP_IGNORE_VALUE_FUNCTION:
        discount_factor = 1
    else:
        discount_factor = human_expectations_value_function.discount_factor

    if resume_from_output_dir is not None:
        loading = True
        loaded_welfare_values_filepath = os.path.join(resume_from_output_dir, WELFARE_VALUES_FILENAME)
        with open(loaded_welfare_values_filepath, 'r') as f:
            welfare_values = json.load(f)
            reward_values_loaded = welfare_values["reward_values"]
            future_value_values_loaded = welfare_values["future_value_values"]
            past_rewards_values_loaded = welfare_values["past_rewards_values"]
            past_rewards_and_future_value_values_loaded = welfare_values["past_rewards_and_future_value_values"]

        reward_values_loaded = {val[0]: val[1:] for val in reward_values_loaded}
        future_value_values_loaded = {val[0]: val[1:] for val in future_value_values_loaded}
        past_rewards_values_loaded = {val[0]: val[1:] for val in past_rewards_values_loaded}
        past_rewards_and_future_value_values_loaded = {val[0]: val[1:] for val in past_rewards_and_future_value_values_loaded}
    else:
        loading = False
    
    current_reward_measure = CurrentRewardMeasure(
        discount_factor=discount_factor,
        short_term_horizon=short_term_horizon,
        agent_policy=expected_agent_policy,
        env=expected_env,
    )
    past_rewards_measure = PastRewardsMeasure(
        discount_factor=discount_factor,
        short_term_horizon=retrospective_length,
        agent_policy=expected_agent_policy,
        env=expected_env,
    )
    future_value_measure = FutureValueMeasure(
        discount_factor=discount_factor,
        short_horizon_before_valuefn=short_term_horizon,
        agent_policy=expected_agent_policy,
        env=expected_env,
        value_function=human_expectations_value_function,
    )


    
    for i, traj_step in enumerate(tqdm(trajectory, desc="Detecting welfare reduction over trajectory steps")):

        state, action, reward, info = traj_step.state, traj_step.action, traj_step.reward, traj_step.info

        if loading and i in reward_values_loaded:
            actual_reward, expected_reward = reward_values_loaded[i]
        else:
            actual_reward = current_reward_measure.measure_from_trajectory(
                whole_trajectory=trajectory,
                current_index=i
            )
            expected_reward = current_reward_measure.monte_carlo_estimate(
                whole_trajectory=trajectory,
                current_index=i,
                num_rollouts=num_mc_rollouts,
            )

        reward_values.append((i, actual_reward, expected_reward))
            
        if not TEMP_IGNORE_VALUE_FUNCTION:

            if loading and i in future_value_values_loaded:
                actual_future_value, expected_future_value = future_value_values_loaded[i]
            else:
                actual_future_value = future_value_measure.measure_from_trajectory(
                    whole_trajectory=trajectory,
                    current_index=i
                )
                expected_future_value = future_value_measure.monte_carlo_estimate(
                    whole_trajectory=trajectory,
                    current_index=i,
                    num_rollouts=num_mc_rollouts,
                )

            future_value_values.append((i, actual_future_value, expected_future_value))
        
        if i > retrospective_length:
            if debugging_retrospective_cutoff is not None:
                if i < len(trajectory) - debugging_retrospective_cutoff:
                    continue

            past_i = i-retrospective_length
            if past_i < 0:
                raise ValueError(f"At step {i}, not enough past states for a retrospective length of {retrospective_length}")
            past_traj_step = trajectory[past_i]
            past_state = past_traj_step.state

            if loading and i in past_rewards_values_loaded:
                actual_past_rewards, expected_past_rewards = past_rewards_values_loaded[i]
            else:
                actual_past_rewards = past_rewards_measure.measure_from_trajectory(
                    whole_trajectory=trajectory,
                    current_index=i
                )
                expected_past_rewards = past_rewards_measure.monte_carlo_estimate(
                    whole_trajectory=trajectory,
                    current_index=i,
                    num_rollouts=num_mc_rollouts,
                )
            past_rewards_values.append((i, actual_past_rewards, expected_past_rewards))

            if not TEMP_IGNORE_VALUE_FUNCTION:
                if loading and i in past_rewards_and_future_value_values_loaded:
                    actual_past_rewards_and_future_value, expected_past_rewards_and_future_value = past_rewards_and_future_value_values_loaded[i]
                else:
                    # get_actual_past_rewards_and_future_value_fn = partial(
                    #     sum_past_rewards_and_future_value,
                    #     past_state=past_state,
                    #     agent_policy=actual_agent_policy,
                    #     env=actual_env,
                    #     num_steps=retrospective_length,
                    #     value_function=human_expectations_value_function
                    # )
                    # get_expected_past_rewards_and_future_value_fn = partial(
                    #     sum_past_rewards_and_future_value,
                    #     past_state=past_state,
                    #     agent_policy=expected_agent_policy,
                    #     env=expected_env,
                    #     num_steps=retrospective_length,
                    #     value_function=human_expectations_value_function
                    # )
                    # actual_past_rewards_and_future_value = mc_welfare2(
                    #     welfare_function=get_actual_past_rewards_and_future_value_fn,
                    #     num_rollouts=num_mc_rollouts
                    # )
                    # expected_past_rewards_and_future_value = mc_welfare2(
                    #     welfare_function=get_expected_past_rewards_and_future_value_fn,
                    #     num_rollouts=num_mc_rollouts
                    # )
                    raise NotImplementedError("This function is hasn't been updated to latest code")
                past_rewards_and_future_value_values.append((i, actual_past_rewards_and_future_value, expected_past_rewards_and_future_value))
    
        if i > 0 and i % 10 == 0:
            welfare_plot_fn(
                reward_values=reward_values,
                future_value_values=future_value_values,
                past_rewards_values=past_rewards_values,
                past_rewards_and_future_value_values=past_rewards_and_future_value_values,
                output_dir=output_dir,
                compare_smoothing=True,
            )

        if i >0 and i % 10 == 0:
            # Save the welfare values every 10 steps, as json
            with open(os.path.join(output_dir, WELFARE_VALUES_FILENAME), 'w') as f:
                json.dump({
                    "reward_values": reward_values,
                    "future_value_values": future_value_values,
                    "past_rewards_values": past_rewards_values,
                    "past_rewards_and_future_value_values": past_rewards_and_future_value_values
                }, f)


    return reward_values, future_value_values, past_rewards_values, past_rewards_and_future_value_values

def detect_welfare_reduction_windows(
    trajectory: 'Trajectory', 
    actual_agent_policy: 'Policy', 
    expected_agent_policy: 'Policy', 
    actual_env: 'EnvRenderWrapper', 
    expected_env: 'EnvRenderWrapper', 
    human_expectations_value_function: ValueFunction, 
    retrospective_length: int, 
    short_term_horizon: int, 
    output_dir: str,
    welfare_plot_fn: callable,
    debugging_retrospective_cutoff: int = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:

    raise RuntimeError("This function is deprecated. Use detect_welfare_reduction() and apply smoothing in visualization function")
    
    if TEMP_IGNORE_VALUE_FUNCTION:
        discount_factor = 1
    else:
        discount_factor = human_expectations_value_function.discount_factor

    # current reward, future value, sum past rewards, sum past rewards and future value
    current_reward_actual = [current_reward(traj_step.state, actual_agent_policy, actual_env, short_term_horizon, discount_factor) for traj_step in trajectory]
    smoothed_current_reward_expected = smoothed_welfare(trajectory, expected_agent_policy, expected_env, current_reward, 2, short_term_horizon, discount_factor)
    reward_values = [(i, current_reward_actual[i], smoothed_current_reward_expected[i]) for i in range(len(trajectory))]
    
    print('We got reward values')

    if not TEMP_IGNORE_VALUE_FUNCTION:
        future_value_actual = [future_value(traj_step.state, actual_agent_policy, actual_env, human_expectations_value_function, short_term_horizon) for traj_step in trajectory]
        smoothed_future_value_expected = smoothed_welfare(trajectory, expected_agent_policy, expected_env, future_value, 2, human_expectations_value_function, short_term_horizon)
        future_value_values = [(i, future_value_actual[i], smoothed_future_value_expected[i]) for i in range(len(trajectory))]
    else:
        future_value_values = []

    print('We got future value values')
    
    new_traj = Trajectory()
    retrospective_cutoff = 0

    if debugging_retrospective_cutoff is not None:
        retrospective_cutoff = debugging_retrospective_cutoff
    else:
        retrospective_cutoff = retrospective_length
    
    t_len = len(trajectory)
    new_traj.steps = trajectory.steps[:t_len-retrospective_cutoff]

    # actual_sum_past_rewards = [sum_past_rewards(traj_step.state, actual_agent_policy, actual_env, discount_factor, retrospective_length) for traj_step in new_traj]
    # smoothed_sum_past_rewards_expected = smoothed_welfare(new_traj, expected_agent_policy, expected_env, sum_past_rewards, 2, discount_factor, retrospective_length)
    # past_rewards_values = [(i, actual_sum_past_rewards[i], smoothed_sum_past_rewards_expected[i]) for i in range(retrospective_cutoff, len(new_traj))]
    past_rewards_values = []

    print('We got past rewards values')

    if not TEMP_IGNORE_VALUE_FUNCTION:
        actual_past_and_future_value = [sum_past_rewards_and_future_value(traj_step.state, actual_agent_policy, actual_env, retrospective_length, human_expectations_value_function) for traj_step in new_traj]
        
        smoothed_past_and_future_value_expected = smoothed_welfare(new_traj,
                                                                    expected_agent_policy,
                                                                    expected_env,
                                                                    sum_past_rewards_and_future_value,
                                                                    2,
                                                                    retrospective_length,
                                                                    human_expectations_value_function)
        
        past_rewards_and_future_value_values = [(i, actual_past_and_future_value[i], smoothed_past_and_future_value_expected[i]) for i in range(retrospective_cutoff, len(new_traj))]
    else:
        past_rewards_and_future_value_values = []

    print('We got past rewards and future value values')
    
    for j in range(0, len(trajectory), 25):
        welfare_plot_fn(
            reward_values=reward_values,
            future_value_values=future_value_values,
            past_rewards_values=past_rewards_values,
            past_rewards_and_future_value_values=past_rewards_and_future_value_values,
            output_dir=output_dir,
        )

    return reward_values, future_value_values, past_rewards_values, past_rewards_and_future_value_values

def generate_kwargs(traj, 
                    actual_agent_policy, 
                    expected_agent_policy, 
                    actual_env, 
                    expected_env, 
                    nao_support_policy,
                    human_expectations_value_function, 
                    retrospective_length_frames, 
                    short_term_horizon_frames,
                    num_mc_rollouts, 
                    output_dir,
                    welfare_plot_fn):
    """ 
    Purpose: Generate kwargs for the detect_welfare_reduction2 method 
    Returns: Dictionary of keyword arguments for detect_welfare_reduction2 method 
    """
    kwargs = {
        'trajectory': traj if traj is not None else None,
        'actual_agent_policy': actual_agent_policy if actual_agent_policy is not None else None,
        'expected_agent_policy': expected_agent_policy if expected_agent_policy is not None else None,
        'actual_env': actual_env if actual_env is not None else None, 
        'expected_env': expected_env if expected_env is not None else None, 
        'nao_support_policy': nao_support_policy if nao_support_policy is not None else NaoSupportPolicies.IDLE,
        'human_expectations_value_function': human_expectations_value_function if human_expectations_value_function is not None else None, 
        'retrospective_length': retrospective_length_frames if retrospective_length_frames is not None else None,
        'short_term_horizon': short_term_horizon_frames if short_term_horizon_frames is not None else None,
        'num_mc_rollouts': num_mc_rollouts if num_mc_rollouts is not None else 10,
        'output_dir': output_dir if output_dir is not None else None,
        'welfare_plot_fn': welfare_plot_fn if welfare_plot_fn is not None else None,
        'resume_from_output_dir': None,
        'debugging_retrospective_cutoff': None
    }
    
    return kwargs

def match_scenario(scenario, json_files, json_dir, mode='shoot'):
    """ 
    Purpose: Find the json file the trajectory of which a user watched in the user study
    Returns: The path of the json file the trajectory of which a user watched, None if no trajectory matches.
    Args: 
        scenario (pandas.Series): a row of a dataframe obtained by calling user_study_df.iterrows()
        json_files (List[str]): a list of strings containing the json_files in json_dir
        json_dir (str): a path to the directory containing json_files
        mode (str): a string specifying which mode the user study watched a video of. For now, assuming shoot. 
    """
    # TODO: Mode in df?
    behavior = scenario['NaoBehavior']
    initial = scenario['startState']
    ending = scenario['endState']

    for json_file in json_files:
        if behavior in json_file and initial in json_file and ending in json_file and mode in json_file:
            return os.path.join(json_dir, json_file)
    
    return None

def interface_data(df, json_dir, welfares_dir, **kwargs):
    """
    Purpose: Generate JSON files containing welfare measure results for each participant.
    Args:
        df (pandas.DataFrame): DataFrame containing participant information and metadata required to match and process
                               the corresponding trajectory files.
        json_dir (str): Directory path where the JSON trajectory files are stored.
        welfares_dir (str): Directory path where the output JSON welfare measure files will be saved.
        **kwargs: Additional keyword arguments to be passed to detect_welfare_reduction2.
    Returns:
        None
    Raises:
        Exception: Propagates any exceptions raised during file operations or while processing the data.
    """ 

    index_to_referent_standard = {1: NaoSupportPolicies.EQUALIZE_SCORES,
                                  2: NaoSupportPolicies.EQUALIZE_SCORES_NEUTRAL,
                                  3: NaoSupportPolicies.HUMAN_ONLY,
                                  4: NaoSupportPolicies.SHUTTER_ONLY,
                                  5: NaoSupportPolicies.IDLE,
                                  6: "Other"}
    
    json_files = os.listdir(json_dir)

    for index, row in df.iterrows():
        referent_standard = index_to_referent_standard[row['MIP_RF']]
        if referent_standard == 'Other':
            # For now, skip if referent standard is other 
            continue 
        
        json_file = match_scenario(row, json_files, json_dir)
        if json_file is None:
            print('no json file matches that of the participant passed in.')
            continue 

        loaded_traj = Trajectory.from_file(json_file)
        kwargs['trajectory'] = loaded_traj
        kwargs['nao_support_policy'] = referent_standard
        nonlocalized_reward_values, single_step_reward_values, complete_past_rewards_values = detect_welfare_reduction2(**kwargs)

        filename = f"Participant_{row['Participant']}_Video_{row['VideoOrder']}_{row['NaoBehavior']}_{row['startState']}_{row['endState']}.json"
        file_path = os.path.join(welfares_dir, welfares_dir, filename)

        with open(file_path, 'w') as f:
            json.dump(
                {
                    "nonlocalized_reward_values": nonlocalized_reward_values,
                    "single_step_reward_values": single_step_reward_values,
                    "complete_past_rewards_values": complete_past_rewards_values,
                    "user_label": row['rw_average'],
                    'expected_policy': row['MIP_RF']
                }, f
        )

def detect_welfare_reduction_monte_carlo(
    trajectories: List['Trajectory'], 
    actual_agent_policy: 'Policy', 
    expected_agent_policy: 'Policy', 
    actual_env: 'EnvRenderWrapper', 
    expected_env: 'EnvRenderWrapper', 
    human_expectations_value_function: ValueFunction, 
    retrospective_length: int, 
    short_term_horizon: int, 
    output_dir: str,
    welfare_plot_fn: callable,
    debugging_retrospective_cutoff: int = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:

    raise RuntimeError("This function is deprecated. Use detect_welfare_reduction() and set the num_mc_rollouts parameter to a value greater than 1")
    
    discount_factor = human_expectations_value_function.discount_factor

    # current reward, future value, sum past rewards, sum past rewards and future value
    mc_reward_actual = mc_welfare(trajectories, actual_agent_policy, actual_env, current_reward, short_term_horizon, discount_factor) 
    mc_reward_expected = mc_welfare(trajectories, expected_agent_policy, expected_env, current_reward, short_term_horizon, discount_factor) 
    reward_values = [(i, mc_reward_actual[i], mc_reward_expected[i]) for i in range(len(mc_reward_expected))]
    
    print('We got reward values')

    mc_value_actual = mc_welfare(trajectories, actual_agent_policy, actual_env, future_value, human_expectations_value_function, short_term_horizon)
    mc_value_expected = mc_welfare(trajectories, expected_agent_policy, expected_env, future_value, human_expectations_value_function, short_term_horizon)
    future_value_values = [(i, mc_value_actual[i], mc_value_expected[i]) for i in range(len(mc_reward_expected))]
    
    print('We got future value values')
    
    new_trajs = [Trajectory() for _ in range(len(trajectories))]
    retrospective_cutoff = 0

    if debugging_retrospective_cutoff is not None:
        retrospective_cutoff = debugging_retrospective_cutoff
    else:
        retrospective_cutoff = retrospective_length
    
    for i, trajectory in enumerate(trajectories):
        t_len = len(trajectory)
        new_trajs[i].steps = trajectory.steps[:t_len-retrospective_cutoff]

    mc_actual_sum_past_rewards = mc_welfare(new_trajs, actual_agent_policy, actual_env, sum_past_rewards, discount_factor, retrospective_length)
    mc_expected_sum_past_rewards = mc_welfare(new_trajs, expected_agent_policy, expected_env, sum_past_rewards, discount_factor, retrospective_length)

    past_rewards_values = [(i, mc_actual_sum_past_rewards[i], mc_expected_sum_past_rewards[i]) for i in range(retrospective_cutoff, len(mc_expected_sum_past_rewards))]

    print('We got past rewards values')

    mc_actual_past_and_future_value = mc_welfare(new_trajs, 
                                                 actual_agent_policy, 
                                                 actual_env, 
                                                 sum_past_rewards_and_future_value, 
                                                 retrospective_length, 
                                                 human_expectations_value_function)
    
    mc_expected_past_and_future_value = mc_welfare(new_trajs, 
                                                   expected_agent_policy, 
                                                   expected_env, 
                                                   sum_past_rewards_and_future_value, 
                                                   retrospective_length, 
                                                   human_expectations_value_function)

    
    past_rewards_and_future_value_values = [(i, mc_actual_past_and_future_value[i], mc_expected_past_and_future_value[i]) for i in range(retrospective_cutoff, len(mc_expected_past_and_future_value))]

    print('We got past rewards and future value values')
 
    welfare_plot_fn(
        reward_values=reward_values,
        future_value_values=future_value_values,
        past_rewards_values=past_rewards_values,
        past_rewards_and_future_value_values=past_rewards_and_future_value_values,
        output_dir=output_dir,
    )

    return reward_values, future_value_values, past_rewards_values, past_rewards_and_future_value_values




