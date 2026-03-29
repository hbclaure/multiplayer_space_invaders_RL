import torch
import numpy as np
import copy 
from consts import DynamicsConsts, InitialStateConsts, Players, RenderConsts
from env_utils import tie_breaker_actions, SpaceInvadersState, boolean_policy_to_index, SpaceInvadersConfig
from typing import Callable, Tuple, Union
import pygame

# AgentAction class:
    # Construct from left, right, shoot booleans
    # function to convert to index
# JointAgentAction class:
    # Construct from left, right, shoot booleans for each agent
    # function to convert to index
# Action class:
    # parent to AgentAction and JointAgentAction
    # every Policy instance returns an Action object



class Action:
    def __init__(self):
        raise NotImplementedError("Action.__init__() must be implemented by subclasses of Action")

    def index(self) -> int:
        raise NotImplementedError("index() must be implemented by subclasses of Action")
    
    def short_str(self) -> str:
        raise NotImplementedError("short_str() must be implemented by subclasses of Action")
    
    def __str__(self) -> str:
        raise NotImplementedError("__str__() must be implemented by subclasses of Action")
    
    def __repr__(self) -> str:
        raise NotImplementedError("__repr__() must be implemented by subclasses of Action")
    
    def to_dict(self) -> dict:
        raise NotImplementedError("to_dict() must be implemented by subclasses of Action")
    
    @classmethod
    def from_dict(cls, action_dict: dict) -> "Action":
        raise NotImplementedError("from_dict() must be implemented by subclasses of Action")

from env_utils import boolean_policy_to_index, tie_breaker_actions
from typing import List

class SingleAgentAction(Action):
    def __init__(self, left: bool, right: bool, shoot: bool):
        self.left = left
        self.right = right
        self.shoot = shoot
    
    def index(self) -> int:
        return boolean_policy_to_index(self.left, self.right, self.shoot)
    
    def short_str(self) -> str:
        actions = [action for action, condition in [("left", self.left), ("right", self.right), ("shoot", self.shoot)] if condition]
        return f"Action({', '.join(actions)})"
    
    def __str__(self) -> str:
        return f"SingleAgentAction(left={self.left}, right={self.right}, shoot={self.shoot})"
    
    def __repr__(self) -> str:
        return f"SingleAgentAction(left={self.left}, right={self.right}, shoot={self.shoot})"

    def tie_breaker(self, inplace: bool = True, random_tiebreak: bool = False) -> "SingleAgentAction":
        left, right, shoot = tie_breaker_actions(self.left, self.right, self.shoot, random_tiebreak=random_tiebreak)
        if inplace:
            self.left = left
            self.right = right
            self.shoot = shoot
            return self
        else:
            return SingleAgentAction(left, right, shoot)
    
    def update_from_keyboard_input(self, pygame_events: List, action_component_to_key: dict):

        action_dict = self.to_dict()
        for event in pygame_events:
            
            # Ignore non-keyboard events
            if event.type not in [pygame.KEYDOWN, pygame.KEYUP]:
                continue
            
            # If key pressed, indicate the corresponding action component is true
            # If key released, indicate the corresponding action component is false
            action_component = action_component_to_key.get(event.key, None)
            if action_component is not None:
                if event.type == pygame.KEYDOWN:
                    action_dict[action_component] = True
                elif event.type == pygame.KEYUP:
                    action_dict[action_component] = False

        # Update the action attributes based on the action_dict
        for action_component, value in action_dict.items():
            setattr(self, action_component, value)
    
    def to_dict(self) -> dict:
        return copy.deepcopy(
            {"left": self.left, "right": self.right, "shoot": self.shoot}    
        )
    
    @classmethod
    def from_dict(cls, action_dict: dict) -> "SingleAgentAction":
        return cls(**action_dict)
    
    def __eq__(self, other):
        if not isinstance(other, SingleAgentAction):
            return False
        return self.left == other.left and self.right == other.right and self.shoot == other.shoot
    
    # def from_dict(self, action_dict: dict) -> "SingleAgentAction":
    #     self.left = action_dict["left"]
    #     self.right = action_dict["right"]
    #     self.shoot = action_dict["shoot"]

# TODO
# class JointAgentAction(Action):
#     def __init__(self, all_player_actions ...)


# note: optimal policy is based on obs, rules policy based on state, but either can convert. Pass both and use the one that works?
class Policy:
    def __call__(self, state: SpaceInvadersState) -> Action:
        raise NotImplementedError("__call__() must be implemented by subclasses of Policy")
    
class SingleAgentPolicy(Policy):
    def __init__(self, agent: 'Players', policy_from_state_fn: Callable[['SpaceInvadersState'], Union['SingleAgentAction', Tuple[bool, bool, bool]]], policy_fn_returns_booleans: bool = True):
        assert agent in Players
        self.agent = agent
        self.policy_fn = policy_from_state_fn
        self.policy_fn_returns_booleans = policy_fn_returns_booleans

    def __call__(self, state: SpaceInvadersState) -> SingleAgentAction:
        if self.policy_fn_returns_booleans:
            left, right, shoot = self.policy_fn(state)
            return SingleAgentAction(left, right, shoot)
        else:
            return self.policy_fn(state)    
    
    def __str__(self) -> str:
        return f"SingleAgentPolicy(agent={self.agent})"
    
    def __repr__(self) -> str:
        return f"SingleAgentPolicy(agent={self.agent})"


# def exploration_policy_fn(observations):
#     # input:
#         # observations: (batch_size x obs_dim) numpy array  xxtensor
#     # returns: 
#         # actions: (batch_size x action_dim) numpy array  xxtensor

#     if observations.ndim == 1:
#         observations = np.expand_dims(observations, axis=0)

#     actions = []
#     for obs in observations:
#         action_booleans = human_rules_based_policy_from_obs(obs)
#         action = boolean_policy_to_index(*action_booleans)
#         actions.append(action)

#     actions = torch.tensor(actions)
#     return actions

def create_exploration_policy_fn(policy_from_obs_fn, num_stacked_observations: int = 1):
    def exploration_policy_fn(observations):
        # input:
            # observations: (batch_size x obs_dim) numpy array  xxtensor
        # returns: 
            # actions: (batch_size x action_dim) numpy array  xxtensor

        if observations.ndim == 1:
            observations = np.expand_dims(observations, axis=0)

        actions = []
        for obs in observations:
            if num_stacked_observations > 1:
                obs_dim = obs.size // num_stacked_observations
                stacked_obs = [obs[i * obs_dim:(i + 1) * obs_dim] for i in range(num_stacked_observations)]
                obs = stacked_obs[-1]
            action_booleans = policy_from_obs_fn(obs)
            action = boolean_policy_to_index(*action_booleans)
            actions.append(action)

        return torch.tensor(actions)

    return exploration_policy_fn

# TODO: better way of managing multiple variations of human policy
def minimal_human_rules_based_policy_from_state(state):
    left, right, shoot = False, False, False
    if state.can_shoot(Players.HUMAN):
        shoot = True
    
    return left, right, shoot

def minimal_human_rules_based_policy_from_obs(observation, config: SpaceInvadersConfig):
    state = SpaceInvadersState.from_observation(observation, config, default_init_all_values=False, minimal_observation=True)
    left, right, shoot = minimal_human_rules_based_policy_from_state(state)
    return left, right, shoot

def minimal_always_shoot_human_rules_based_policy_from_state(state):
    left, right, shoot = False, False, True
    return left, right, shoot

def minimal_always_shoot_human_rules_based_policy_from_obs(observation, config: SpaceInvadersConfig):
    state = SpaceInvadersState.from_observation(observation, config, default_init_all_values=False, minimal_observation=True)
    left, right, shoot = minimal_always_shoot_human_rules_based_policy_from_state(state)
    return left, right, shoot

def human_rules_based_policy_from_obs(observation, config: SpaceInvadersConfig):
    state = SpaceInvadersState.from_observation(observation, config, default_init_all_values=False)
    left, right, shoot = human_rules_based_policy_from_state(state)
    
    return left, right, shoot

def human_rules_based_policy_from_state(state):
                
    """
    Policy for human. Identifies nearest bullets and enemies and adjusts its position or shoots.
    """
    DODGE_PROBABILITY = 0


    nearest_enemy_nao = state.get_nearest_enemy(player=Players.NAO, side_to_search='both', excluded_enemies=[])

    # TODO: parse these variables from observation. For now, erroneously using default initialization
    # Missing variables
        # bullets_left_side
        # last_shot_time['Human']

    #initialize variables for actions
    left = False
    right = False
    shoot = False
    hit = False
    nearest_bullet = [0,0]

    #Only focus on bullets on left side
    bullets_left_side, bullets_right_side = state.get_enemy_bullets_by_side()
    bullets_to_search = bullets_left_side

    #Find Nearest bullet to human Ship
    x_diff_prev= 1 #RenderConsts.SCREEN_WIDTH/RenderConsts.SCREEN_WIDTH
    for bullet in bullets_to_search:
        x_diff = abs(bullet[0]-state.players_state.human_position_x)
        #if bullet[1]< InitialStateConsts.HUMAN_POSITION_Y and bullet[1]> DynamicsConsts.VERTICAL_BUFFER and x_diff<DynamicsConsts.HIT_RANGE *2:
        if bullet[1]> DynamicsConsts.VERTICAL_BUFFER and x_diff<DynamicsConsts.HIT_RANGE *2:
            if x_diff < x_diff_prev:
                nearest_bullet= bullet
                x_diff_prev = x_diff

    #look through active enemies and assign them to the left and right side
    # state.enemy_state.active_enemies_right_side.clear()
    # state.enemy_state.active_enemies_left_side.clear()
    # for (x, y), active in zip(zip(InitialStateConsts.ENEMIES_X, InitialStateConsts.ENEMIES_Y), state.enemy_state.enemies_active):
    #     if active:
    #         if x >= 0.5:
    #             state.enemy_state.active_enemies_right_side.append((x,y))
    #         else:
    #             state.enemy_state.active_enemies_left_side.append((x,y))
    active_enemies_left_side, active_enemies_right_side = state.get_active_enemies_by_side()

    #only search through enemies on left side
    enemies_to_search = active_enemies_left_side

    # # # find nearest enemy by Y
    # nearest_enemy = [0,0]
    # nearest_x_diff = 1 #RenderConsts.SCREEN_WIDTH
    # nearest_y_diff = DynamicsConsts.VERTICAL_BUFFER
    # closest_x_diff =1 # RenderConsts.SCREEN_WIDTH



    # #In the Js version they looked through each enemy and checked if it was active or not. We are just going to check active enmies. 
    # #Find nearest enemy to Nao Ship
    # for enemy in enemies_to_search:
    #     if enemy == nearest_enemy_nao:
    #         pass
    #     else:
    #         check_distance_x = abs(enemy[0] - state.players_state.human_position_x)
    #         check_distance_y = abs(enemy[1] - InitialStateConsts.HUMAN_POSITION_Y)
    #         if check_distance_y < nearest_y_diff:
    #             nearest_enemy = enemy
    #             nearest_x_diff = check_distance_x
    #             nearest_y_diff = check_distance_y
    #         elif check_distance_y == nearest_y_diff and check_distance_x < nearest_x_diff:
    #             nearest_enemy = enemy
    #             nearest_x_diff = check_distance_x
    #             nearest_y_diff = check_distance_y
    #         elif check_distance_y < (nearest_y_diff+(50/RenderConsts.SCREEN_HEIGHT)) and check_distance_x < closest_x_diff :
    #             closest_x_diff = check_distance_x

    nearest_enemy = state.get_nearest_enemy(
        player=Players.HUMAN, side_to_search='left', excluded_enemies=[]

        #player=Players.HUMAN, side_to_search='left', excluded_enemies=[nearest_enemy_nao]
    )


    # check if ai_agent is in danger of being hit by bullet
    if nearest_bullet[0] <= state.players_state.human_position_x + DynamicsConsts.HIT_RANGE and nearest_bullet[0] >= state.players_state.human_position_x - DynamicsConsts.HIT_RANGE:
        hit = True

    # Introduce probabilistic dodging

    approached_enemy = False


    if not state.can_shoot(Players.HUMAN):
        if hit:
            if nearest_bullet[0] >= 1 - (75 / RenderConsts.SCREEN_WIDTH):
                left = True #False #True
            elif nearest_bullet[0] <= 55 / RenderConsts.SCREEN_WIDTH:
                right = True #False #True
            elif nearest_bullet[0] > state.players_state.human_position_x:
                left = True # False #True
            elif nearest_bullet[0] <= state.players_state.human_position_x:
                right = True #False #True
    else:
        # TODO: Why not include this in the can_shoot logic?
        if abs(nearest_enemy[0] - state.players_state.human_position_x) <= DynamicsConsts.SHOOTING_RANGE:
            shoot = True

        if nearest_enemy[0] < state.players_state.human_position_x:
            if not (nearest_bullet[0] < state.players_state.human_position_x and InitialStateConsts.HUMAN_POSITION_Y - nearest_bullet[1] < (200 / RenderConsts.SCREEN_HEIGHT) and state.players_state.human_position_x - nearest_bullet[0] <= DynamicsConsts.SECOND_HIT_RANGE):
                left = True
                approached_enemy = True
        elif nearest_enemy[0] > state.players_state.human_position_x:
            if not (nearest_bullet[0] > state.players_state.human_position_x and InitialStateConsts.HUMAN_POSITION_Y - nearest_bullet[1] < (200 / RenderConsts.SCREEN_HEIGHT) and nearest_bullet[0] - state.players_state.human_position_x <= DynamicsConsts.SECOND_HIT_RANGE):
                right = True
                approached_enemy = True

        if not approached_enemy and hit:
            if abs(nearest_bullet[0] - state.players_state.human_position_x) > 5 / RenderConsts.SCREEN_WIDTH:
                shoot = False
            # TODO: what's 75 and 55?
            if nearest_bullet[0] >= 1 - (75 / RenderConsts.SCREEN_WIDTH):
                left = True
            elif nearest_bullet[0] <= 55 / RenderConsts.SCREEN_WIDTH:
                right = True
            elif nearest_bullet[0] > state.players_state.human_position_x:
                left = True
            elif nearest_bullet[0] <= state.players_state.human_position_x:
                right = True

    left, right, shoot = tie_breaker_actions(left, right, shoot)

    assert sum([left, right, shoot]) <= 1, "Agent can only execute one action at a time"
    
    # if shoot:
    #     randomize_shooting = random.randint(0,1)
    #     print('randomize shooting',randomize_shooting)
    #     if randomize_shooting ==0:
    #         print('here')
    #         shoot = False
    # print('shoot2',shoot)
    return left,right, shoot
