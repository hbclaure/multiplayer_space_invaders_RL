import pygame
import os
from dataclasses import dataclass, field, fields
import numpy as np
import time
import copy
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from consts import RenderConsts, Players, Enemies, StateConsts, InitialStateConsts, DynamicsConsts, ObservationConsts, ActionConsts
from consts import RewardTypes, RenderModes, NaoSupportPolicies, ActionSpaces, PolicyConsts, InteractiveModes, GameTypes, ScoreConsts
from units_utils import seconds_to_frames as _seconds_to_frames
from units_utils import frames_to_seconds as _frames_to_seconds

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agents.robot_integration import RobotActionRequest
    from interactive_gameplay.utils import GameScriptEvent

def load_image(filename, shape, opacity=None):
    # shape is (width, height)
    image = pygame.image.load(os.path.join(RenderConsts.IMAGE_DIR, filename))
    image = pygame.transform.scale(image, shape)
    if opacity is not None:
        # set the opacity of the image
        image.set_alpha(opacity)

    return image

def load_ship_image(ship_name, inactive=False):
    if ship_name == Players.HUMAN.value:
        image_filename = RenderConsts.HUMAN_SHIP_IMAGE
    elif ship_name == Players.SHUTTER.value:
        image_filename = RenderConsts.SHUTTER_SHIP_IMAGE
    elif ship_name == Players.NAO.value:
        if PolicyConsts.POLICY_MODE == 'shooting':
            image_filename = RenderConsts.NAO_SHIP_IMAGE
        elif PolicyConsts.POLICY_MODE == 'powerUp':
            image_filename = RenderConsts.NAO_POWERUP_IMAGE
    elif ship_name == Enemies.ENEMY1.value:
        image_filename = RenderConsts.ENEMY_IMAGE1
    elif ship_name == Enemies.ENEMY2.value:
        image_filename = RenderConsts.ENEMY_IMAGE2
    else:
        raise ValueError(f"Invalid ship name: {ship_name}")

    opacity = 255//4 if inactive else None # 1/4 opacity for inactive ships
    return load_image(image_filename, (RenderConsts.SHIP_WIDTH, RenderConsts.SHIP_HEIGHT), opacity=opacity)

def load_bullet_image(enemy=False):
    if enemy:
        image_filename = RenderConsts.ENEMY_BULLET_IMAGE
    else:
        image_filename = RenderConsts.BULLET_IMAGE

    return load_image(image_filename, (RenderConsts.BULLET_WIDTH, RenderConsts.BULLET_HEIGHT))

def boolean_policy_to_onehot(left, right, shoot):
    assert left + right + shoot <= 1
    action = [0, 0, 0, 0]
    if left:
        action[0] = 1
    if right:
        action[1] = 1
    if shoot:
        action[2] = 1
    return np.array(action)

def boolean_policy_to_index(left, right, shoot):
    assert left + right + shoot <= 1

    if left:
        return 0
    elif right:
        return 1
    elif shoot:
        return 2
    else:
        return 3
    
def index_to_boolean_policy(index):
    assert index in range(ActionConsts.DIM)
    if index == 0:
        return True, False, False
    elif index == 1:
        return False, True, False
    elif index == 2:
        return False, False, True
    else:
        return False, False, False
    
def tie_breaker_actions(left_movement, right_movement, shoot, random_tiebreak=False):
    """
    Randomly disable one action if both left and shoot are true.
    """
    if left_movement and right_movement:
        # don't move either direction
        left_movement = False
        right_movement = False

    if random_tiebreak:
        #randomly decide between one movement in cases where you can move and shoot
        if left_movement and shoot:
            if random.choice([True, False]):
                left_movement = False
            else:
                shoot = False

        elif right_movement and shoot:
            if random.choice([True, False]):
                right_movement = False
            else:
                shoot = False
        return left_movement, right_movement, shoot
    else:
        # always shoot if shooting and moving is desired: can move in the next frame
        if left_movement and shoot:
            left_movement = False
        elif right_movement and shoot:
            right_movement = False
        return left_movement, right_movement, shoot

def frames_to_seconds(num_frames):
    return _frames_to_seconds(num_frames, DynamicsConsts.FRAMES_PER_SECOND)

def seconds_to_frames(num_seconds):
    return _seconds_to_frames(num_seconds, DynamicsConsts.FRAMES_PER_SECOND)

'''
# One way to include bounds enforcement
def validate_bounds(instance):
    for f in fields(instance):
        bounds = f.metadata.get('bounds', None)
        value = getattr(instance, f.name)
        if bounds and value is not None:
            min_val, max_val = bounds
            if not (min_val <= value <= max_val):
                raise ValueError(f"{f.name}={value} is out of bounds ({min_val}, {max_val})")

@dataclass
class AllPlayersState:
    human_position_x: float = field(default=None, metadata={'bounds': (0.0, 100.0)})
    human_position_y: float = field(default=None, metadata={'bounds': (0.0, 100.0)})
    shutter_position_x: float = field(default=None, metadata={'bounds': (0.0, 100.0)})
    shutter_position_y: float = field(default=None, metadata={'bounds': (0.0, 100.0)})
    nao_position_x: float = field(default=None, metadata={'bounds': (0.0, 100.0)})
    nao_position_y: float = field(default=None, metadata={'bounds': (0.0, 100.0)})
    human_shoots: bool = None
    shutter_shoots: bool = None
    nao_shoots: bool = None
    active: Dict[str, bool] = None

    support: int = field(default=None, metadata={'bounds': (0, 10)})

    spaceship_left_edge: float = field(default=None, metadata={'bounds': (0.0, 100.0)})
    spaceship_right_edge: float = field(default=None, metadata={'bounds': (0.0, 100.0)})
    spaceship_upper_edge: float = field(default=None, metadata={'bounds': (0.0, 100.0)})

    def validate(self):
        validate_bounds(self)
    
    def __post_init__(self, require_all_attributes=True):
        if require_all_attributes:
            for f in fields(self):
                if getattr(self, f.name) is None:
                    raise ValueError(f"Missing attribute: {f.name}")
        self.validate()
'''

class StrictDataclass:
    _init_done = False
    def __post_init__(self):
        # In the case of frozen dataclasses, __setattr__ is required to workaround the frozen restriction
        object.__setattr__(self, '_field_names', {f.name for f in fields(self)})
        object.__setattr__(self, '_init_done', True)

    def _custom_compare_field(self, field_name, self_value, other_value):
        # Default implementation, can be overridden by subclasses
        return None
    
    def equal(self, other, ignore_missing_fields=False, enforce_type_match=False):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot compare {self.__class__.__name__}")
        
        for f in fields(self):
            self_value = getattr(self, f.name)
            other_value = getattr(other, f.name)

            if ignore_missing_fields and (self_value is None or other_value is None):
                continue

            if enforce_type_match and type(self_value) != type(other_value):
                raise ValueError(f"Type mismatch for field {f.name}: {type(self_value)} != {type(other_value)}")
            
            # Call the hook method for special cases
            custom_compare = self._custom_compare_field(f.name, self_value, other_value)
            if custom_compare is not None:
                if not custom_compare:
                    return False
                continue            
            
            # if variables are iterable, compare element-wise
            if isinstance(self_value, (list, tuple)):
                if not all(x == y for x, y in zip(self_value, other_value)):
                    if not all(np.allclose(x, y, atol=1e-3) for x, y in zip(self_value, other_value)):
                        return False
            elif isinstance(self_value, np.ndarray):
                if not np.array_equal(self_value, other_value):
                    if not np.allclose(self_value, other_value, atol=1e-3):
                        return False
            # if variables are dictionaries, compare element-wise
            # NOTE: short-circuiting with != allows comparison of None values that np.allclose does not support
            elif isinstance(self_value, dict):
                if len(self_value) != len(other_value):
                    return False
                for key in self_value:
                    if self_value[key] != other_value[key]:
                        if not np.allclose(self_value[key], other_value[key], atol=1e-3):
                            return False
            else:
                if self_value != other_value:
                    return False
        return True
    
    # Override __setattr__ to enforce strict attribute setting
    def __setattr__(self, name, value):
        if not self._init_done:
            super().__setattr__(name, value)
            return
        if name not in self._field_names:
            raise AttributeError(f"Cannot set attribute {name}, it is not among the fields of the class: {self._field_names}")
        super().__setattr__(name, value)

    # def to_dict(self):
    #     return {field_name : getattr(self, field_name) for field_name in self._field_names}
    
    # @classmethod
    # def from_dict(cls, dct, legacy_file_structure=False):
    #     return cls(**dct)

    def to_dict(self):
        dct = {field_name: getattr(self, field_name) for field_name in self._field_names}

        # json doesn't play nice with the Players enum, so replace with string names
        for field_name, value in dct.items():
            if isinstance(value, dict):
                new_dict = {}
                for k, v in value.items():
                    if isinstance(k, Players):
                        new_dict[k.name] = v
                    else:
                        new_dict[k] = v
                dct[field_name] = new_dict

        return dct
    
    @classmethod
    def from_dict(cls, dct, legacy_file_structure=False):
        obj = cls(**dct)

        # recreate enums from string names
        for field_name, value in dct.items():
            if isinstance(value, dict):
                new_dict = {}
                for k, v in value.items():
                    try:
                        new_dict[Players[k]] = v
                    except KeyError:
                        new_dict[k] = v
                setattr(obj, field_name, new_dict)

        return obj

@dataclass
class AllPlayersState(StrictDataclass):
    human_position_x: float = None
    human_position_y: float = None
    shutter_position_x: float = None
    shutter_position_y: float = None
    nao_position_x: float = None
    nao_position_y: float = None
    active: Dict[str, bool] = None
    nao_supporting_player: Players = None
    double_bullet_boost: Dict[Players, bool] = None

    def validate(self):
        pass
        # TODO: enforce bounds and internal consistency rules

    # def to_dict(self):
    #     return {
    #         'human_position_x': self.human_position_x,
    #         'human_position_y': self.human_position_y,
    #         'shutter_position_x': self.shutter_position_x,
    #         'shutter_position_y': self.shutter_position_y,
    #         'nao_position_x': self.nao_position_x,
    #         'nao_position_y': self.nao_position_y,
    #         'active': self.active
    #     }
    
    # def from_dict(self, dct):
    #     self.human_position_x = dct['human_position_x']
    #     self.human_position_y = dct['human_position_y']
    #     self.shutter_position_x = dct['shutter_position_x']
    #     self.shutter_position_y = dct['shutter_position_y']
    #     self.nao_position_x = dct['nao_position_x']
    #     self.nao_position_y = dct['nao_position_y']
    #     self.active = dct['active']


    # NOTE: these two functions are not needed anymore due to CustomJSONEncoder handling Enums    
    # def to_dict(self):
    #     dct = super().to_dict()
    #     dct['nao_supporting_player'] = self.nao_supporting_player.name
    #     return dct

    # @classmethod
    # def from_dict(cls, dct, legacy_file_structure=False):
    #     obj = super().from_dict(dct)

    #     if legacy_file_structure:
    #         obj.nao_supporting_player = None
    #     else:
    #         obj.nao_supporting_player = Players[dct['nao_supporting_player']]
    #     return obj



@dataclass
class EnemyState(StrictDataclass):
    enemies_active: List[bool] = None
    enemies_hit_frames: List[int] = None

    # def to_dict(self):
    #     return {
    #         'enemies_active': self.enemies_active,
    #         'enemies_hit_frames': self.enemies_hit_frames
    #     }
    
    # def from_dict(self, dct):
    #     self.enemies_active = dct['enemies_active']
    #     self.enemies_hit_frames = dct['enemies_hit_frames']

@dataclass
class BulletState(StrictDataclass):
    enemy_bullets: np.ndarray = None # shape num bullets by 2d
    player_bullets: np.ndarray = None
    bullet_y_positions: np.ndarray = None
    shoot_counter: int = None
    player_multi_bullet_counts: np.ndarray = None

    enemyb_free: List[int] = None
    enemyb_used: List[int] = None
    playerb_free: List[int] = None
    playerb_used: List[int] = None

    human_distance_to_bullets: List[float] = None
    small_distance_to_bullets: List[float] = None
    bullet_threat_binary: List[bool] = None
    frames_until_collision: List[float] = None
    shutter_distance_to_bullets: List[float] = None
    shutter_small_distance_to_bullets: List[float] = None
    shutter_bullet_threat_binary: List[bool] = None
    shutter_frames_until_collision: List[float] = None

    def _replace_none_with_nan(self, array):
        return np.where(array == None, np.nan, array) 
    
    def _custom_compare_field(self, field_name, self_value, other_value):
        if field_name == 'enemy_bullets' or field_name == 'player_bullets':
            # enemy_bullets is a 2D array, where order of the rows is arbitrary
            # so we need to compare rows element-wise agnostic to order
            
            if self_value.shape != other_value.shape:
                return False
            
            # Deep copy to avoid modifying the original arrays
            self_value = copy.deepcopy(self_value)
            other_value = copy.deepcopy(other_value)

            # Replace string player names with indices
            if field_name == 'player_bullets':
                for i in range(self_value.shape[0]):
                    self_shooter = self_value[i, 2]
                    if self_shooter is not None:
                        self_value[i, 2] = Players(self_shooter).index()
                    other_shooter = other_value[i, 2]
                    if other_shooter is not None:
                        other_value[i, 2] = Players(other_shooter).index()
                    
            
            # Replace None with np.nan
            self_value = self._replace_none_with_nan(self_value)
            other_value = self._replace_none_with_nan(other_value)
            
            # Sort the rows of both arrays
            sorted_self_value = np.sort(self_value.astype(float), axis=0)
            sorted_other_value = np.sort(other_value.astype(float), axis=0)
            
            # Compare the sorted arrays
            is_close = np.allclose(sorted_self_value, sorted_other_value, atol=1e-2, equal_nan=True)
            return is_close
        elif field_name == 'enemyb_free' or field_name == 'enemyb_used' or field_name == 'playerb_free' or field_name == 'playerb_used':
            # which indexes are free or used is arbitrary, all that matters is how many
            return len(self_value) == len(other_value)
        # call super method for other fields to return default value
        return super()._custom_compare_field(field_name, self_value, other_value)

@dataclass
class ScoreState(StrictDataclass):
    scores: Dict[str, float] = None
    # human_hit_count: int = None # DEPRECATED

@dataclass
class TimeState(StrictDataclass):
    frame: int = None
    stamp: float = None

@dataclass
class RewardState(StrictDataclass):
    sum_reward: float = None


@dataclass
class History(StrictDataclass):
    # These are shooting states, might be better in Player/Enemy State?
    average_shot_frequency_frames: Dict[str, int] = None

    # Cumulative frames of Nao support, indexed by Player
    support_frame_count: Dict[Players, float] = None
    
    # These could belong to TimeState?
    shot_frame_history: Dict[str, list] = None
    last_shot_frame: Dict[str, int] = None
    last_hit_frame: Dict[str, int] = None
    last_enemy_shot_frame: int = None

    def to_dict(self):
        # json doesn't play nice with the Players enum
        # return copy.deepcopy({
        #     'average_shot_frequency_frames': self.average_shot_frequency_frames,
        #     'human_support_frame_count': self.support_frame_count[Players.HUMAN],
        #     'shutter_support_frame_count': self.support_frame_count[Players.SHUTTER],
        #     'nao_support_frame_count': self.support_frame_count[Players.NAO],
        #     'shot_frame_history': self.shot_frame_history,
        #     'last_shot_frame': self.last_shot_frame,
        #     'last_hit_frame': self.last_hit_frame,
        #     'last_enemy_shot_frame': self.last_enemy_shot_frame
        # })

        dct = super().to_dict()

        # json doesn't play nice with the Players enum, so replace with string names
        # dct['support_frame_count'] = {
        #     player.name: count for player, count in self.support_frame_count.items()
        # }
        return dct
    
    # def from_dict_old(self, dct):
    #     self.average_shot_frequency_frames = dct['average_shot_frequency_frames']
    #     self.support_frame_count = {
    #         Players.HUMAN: dct['human_support_frame_count'],
    #         Players.SHUTTER: dct['shutter_support_frame_count'],
    #         Players.NAO: dct['nao_support_frame_count']
    #     }
    #     self.shot_frame_history = dct['shot_frame_history']
    #     self.last_shot_frame = dct['last_shot_frame']
    #     self.last_hit_frame = dct['last_hit_frame']
    #     self.last_enemy_shot_frame = dct['last_enemy_shot_frame']

    @classmethod
    def from_dict(cls, dct, legacy_file_structure=False):
        if legacy_file_structure:
            dct["support_frame_count"] = {
                Players.HUMAN: dct["human_support_frame_count"],
                Players.SHUTTER: dct["shutter_support_frame_count"],
                Players.NAO: dct["nao_support_frame_count"]
            }
            del dct["human_support_frame_count"]
            del dct["shutter_support_frame_count"]
            del dct["nao_support_frame_count"]
            return super().from_dict(dct)
        else:
        
            obj = super().from_dict(dct)

            # json doesn't play nice with the Players enum, so replace with string names
            # obj.support_frame_count = {
            #     Players[player_name]: count for player_name, count in dct['support_frame_count'].items()
            # }
            return obj

    def clear(self):
        self.average_shot_frequency_frames = {'Human': None, 'Shutter': None, 'Nao': None}
        self.support_frame_count = {Players.HUMAN: 0, Players.SHUTTER: 0, Players.NAO: 0}
        self.shot_frame_history = {'Human': [], 'Shutter': [], 'Nao': []}
        self.last_shot_frame = {'Human': float('-inf'), 'Shutter': float('-inf'), 'Nao': float('-inf')}
        self.last_hit_frame = {'Human': float('-inf'), 'Shutter': float('-inf'), 'Nao': float('-inf')}
        self.last_enemy_shot_frame = 0

#from agents.policies import SingleAgentAction
# TODO: fix circular imports
@dataclass
class StepInfo(StrictDataclass):
    # Rewards indexed by player
    enemy_elimination_rewards: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    player_hit_rewards: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    player_shoot_rewards: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    player_taking_score_lead_rewards: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    player_victory_rewards: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))

    # Scoring changes indexed by player
    score_changes: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    
    # Events
    players_hit: List[Players] = field(default_factory=list)
    players_respawned: List[Players] = field(default_factory=list)
    #enemies_hit: List[int] = field(default_factory=list)

    # Actions by all agents (including those in the environment)
    actions_taken: Dict[Players, Any] = field(default_factory=dict)

    # Requested robot actions
    #requested_robot_action: RobotActionRequest = field(default_factory=RobotActionRequest)
    requested_robot_action: Optional['RobotActionRequest'] = field(default=None)

    # NOTE: added after initial implementation/data recording from user study:
    game_script_events: List['GameScriptEvent'] = field(default=None)

    def to_dict(self):
        
        dct = super().to_dict()

        # for attr in dct:
        #     if isinstance(dct[attr], dict):
        #         new_dict = {}
        #         for k, v in dct[attr].items():
        #             if isinstance(k, Players):
        #                 new_dict[k.name] = v
        #             else:
        #                 new_dict[k] = v
        #         dct[attr] = new_dict
        
        return dct
        
        #TODO: there's probably a more flexible way of writing this by just iterating over the fields/attributes conditionally
        #      like, check if attribute is a list or dict, then decompose that way, but this is fine for now.
        return copy.deepcopy({
            # Enemy elimination rewards
            'human_enemy_elimination_rewards': self.enemy_elimination_rewards[Players.HUMAN],
            'shutter_enemy_elimination_rewards': self.enemy_elimination_rewards[Players.SHUTTER],
            'nao_enemy_elimination_rewards': self.enemy_elimination_rewards[Players.NAO],
            # Player hit rewards
            'human_player_hit_rewards': self.player_hit_rewards[Players.HUMAN],
            'shutter_player_hit_rewards': self.player_hit_rewards[Players.SHUTTER],
            'nao_player_hit_rewards': self.player_hit_rewards[Players.NAO],
            # Player shoot rewards
            'human_player_shoot_rewards': self.player_shoot_rewards[Players.HUMAN],
            'shutter_player_shoot_rewards': self.player_shoot_rewards[Players.SHUTTER],
            'nao_player_shoot_rewards': self.player_shoot_rewards[Players.NAO],
            # Player taking score lead rewards
            'human_player_taking_score_lead_rewards': self.player_taking_score_lead_rewards[Players.HUMAN],
            'shutter_player_taking_score_lead_rewards': self.player_taking_score_lead_rewards[Players.SHUTTER],
            'nao_player_taking_score_lead_rewards': self.player_taking_score_lead_rewards[Players.NAO],
            # Player victory rewards
            'human_player_victory_rewards': self.player_victory_rewards[Players.HUMAN],
            'shutter_player_victory_rewards': self.player_victory_rewards[Players.SHUTTER],
            'nao_player_victory_rewards': self.player_victory_rewards[Players.NAO],
            # Score changes
            'human_score_changes': self.score_changes[Players.HUMAN],
            'shutter_score_changes': self.score_changes[Players.SHUTTER],
            'nao_score_changes': self.score_changes[Players.NAO],
            # Players hit
            "human_players_hit": 1 if Players.HUMAN in self.players_hit else 0,
            "shutter_players_hit": 1 if Players.SHUTTER in self.players_hit else 0,
            "nao_players_hit": 1 if Players.NAO in self.players_hit else 0,
            # Players respawned
            "human_players_respawned": 1 if Players.HUMAN in self.players_respawned else 0,
            "shutter_players_respawned": 1 if Players.SHUTTER in self.players_respawned else 0,
            "nao_players_hit_respawned": 1 if Players.NAO in self.players_respawned else 0,
            # Actions taken 
            #NOTE: not sure if these always SingleAction objects... might be Nones?
            'human_actions_taken': self.actions_taken[Players.HUMAN].to_dict(),
            'shutter_actions_taken': self.actions_taken[Players.SHUTTER].to_dict(),
            'nao_actions_taken': self.actions_taken[Players.NAO].to_dict()
        })
    
    def __eq___(self, other):
        return self.to_dict() == other.to_dict()
    
    @classmethod
    def from_dict(cls, dct):
        obj = super().from_dict(dct)

        # for attr in dct:
        #     if isinstance(dct[attr], dict):
        #         new_dict = {}
        #         for k, v in dct[attr].items():
        #             try:
        #                 new_dict[Players[k]] = v
        #             except KeyError:
        #                 new_dict[k] = v
        #         setattr(obj, attr, new_dict)

        return obj
    
    def _from_dict(self, dct):

        # deprecated
        raise NotImplementedError("This method is deprecated, use from_dict instead")

        self.enemy_elimination_rewards = {
            Players.HUMAN: dct['human_enemy_elimination_rewards'],
            Players.SHUTTER: dct['shutter_enemy_elimination_rewards'],
            Players.NAO: dct['nao_enemy_elimination_rewards']
        }
        self.player_hit_rewards = {
            Players.HUMAN: dct['human_player_hit_rewards'],
            Players.SHUTTER: dct['shutter_player_hit_rewards'],
            Players.NAO: dct['nao_player_hit_rewards']
        }
        self.player_shoot_rewards = {
            Players.HUMAN: dct['human_player_shoot_rewards'],
            Players.SHUTTER: dct['shutter_player_shoot_rewards'],
            Players.NAO: dct['nao_player_shoot_rewards']
        }
        self.player_taking_score_lead_rewards = {
            Players.HUMAN: dct['human_player_taking_score_lead_rewards'],
            Players.SHUTTER: dct['shutter_player_taking_score_lead_rewards'],
            Players.NAO: dct['nao_player_taking_score_lead_rewards']
        }
        self.player_victory_rewards = {
            Players.HUMAN: dct['human_player_victory_rewards'],
            Players.SHUTTER: dct['shutter_player_victory_rewards'],
            Players.NAO: dct['nao_player_victory_rewards']
        }
        self.score_changes = {
            Players.HUMAN: dct['human_score_changes'],
            Players.SHUTTER: dct['shutter_score_changes'],
            Players.NAO: dct['nao_score_changes']
        }
        self.players_hit = []
        if dct["human_players_hit"]:
            self.players_hit.append(Players.HUMAN)
        if dct["shutter_players_hit"]:
            self.players_hit.append(Players.SHUTTER)
        if dct["nao_players_hit"]:
            self.players_hit.append(Players.NAO)
        self.players_respawned = []
        if dct["human_players_respawned"]:
            self.players_respawned.append(Players.HUMAN)
        if dct["shutter_players_respawned"]:
            self.players_respawned.append(Players.SHUTTER)
        if dct["nao_players_hit_respawned"]:
            self.players_respawned.append(Players.NAO)
        self.actions_taken = {
            Players.HUMAN: dct['human_actions_taken'],
            Players.SHUTTER: dct['shutter_actions_taken'],
            Players.NAO: dct['nao_actions_taken']
        }

class SpaceInvadersState:

    #start_time: float

    def __init__(self, config: 'SpaceInvadersConfig' = None, init_all_values: bool = False):

        # NOTE: this config is not meant to be saved with every state, and is a lazy shorthand for
        # passing in the config with every call to get_observation()
        self.config = config

        self.players_state = AllPlayersState()
        self.enemy_state = EnemyState()
        self.bullet_state = BulletState()
        self.score_state = ScoreState()
        self.time_state = TimeState()
        self.reward_state = RewardState()
        self.history = History()
        # self.reset()
        # self.validate()

        if init_all_values:
            self.reset()
            self.validate()
    
    #TODO: should properly do this... 
    # def __eq__(self, other):
    #     psEq = self.players_state == other.players_state
    #     esEq = self.enemy_state == other.enemy_state
    #     bsEq = np.array_equal(self.bullet_state, other.bullet_state, equal_nan=True) # just cause sometimes we have nones
    #     ssEq = self.score_state == other.score_state
    #     tsEq = self.time_state == other.time_state
    #     rsEq = self.reward_state == other.reward_state
    #     hsEq = self.history == other.history

    #     return psEq and esEq and bsEq and ssEq and tsEq and rsEq and hsEq

    def reset(self):
        self.players_state = AllPlayersState(
            human_position_x=InitialStateConsts.HUMAN_POSITION_X,
            human_position_y=InitialStateConsts.HUMAN_POSITION_Y,
            shutter_position_x=InitialStateConsts.SHUTTER_POSITION_X,
            shutter_position_y=InitialStateConsts.SHUTTER_POSITION_Y,
            nao_position_x=InitialStateConsts.NAO_POSITION_X,
            nao_position_y=InitialStateConsts.NAO_POSITION_Y,
            active={'Human': True, 'Shutter': True, 'Nao': True},
            nao_supporting_player=Players.NAO, # implies not supporting either player, as None means the field is missing
            double_bullet_boost={player: False for player in Players},
        )
        self.enemy_state = EnemyState(
            enemies_active=[True] * StateConsts.NUM_ENEMIES,
            enemies_hit_frames=[-1] * StateConsts.NUM_ENEMIES
        )
        self.bullet_state = BulletState(
            enemy_bullets=np.array([[0.,0.]] * StateConsts.MAX_NUM_ENEMY_BULLETS),
            player_bullets=np.array([[0.,0.,None]]*StateConsts.MAX_NUM_PLAYER_BULLETS),
            bullet_y_positions=np.zeros((StateConsts.NUM_ENEMY_COLUMNS, StateConsts.MAX_NUM_ENEMY_BULLETS_PER_COLUMN)),
            shoot_counter=0,
            player_multi_bullet_counts=np.zeros(StateConsts.MAX_NUM_PLAYER_BULLETS, dtype=int),
            enemyb_free=list(range(StateConsts.MAX_NUM_ENEMY_BULLETS)),
            enemyb_used=[],
            playerb_free=list(range(StateConsts.MAX_NUM_PLAYER_BULLETS)),
            playerb_used=[],
            human_distance_to_bullets=[0] * StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE * StateConsts.MAX_NUM_ENEMY_BULLETS_PER_COLUMN,
            small_distance_to_bullets=[0]* StateConsts.PRACTICAL_MAX_NUM_ENEMY_BULLETS_PER_SIDE,
            bullet_threat_binary=[0] * StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE,
            frames_until_collision=[float(InitialStateConsts.FRAMES_UNTIL_BULLET_COLLISION)] * StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE,
            shutter_distance_to_bullets=[0] * StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE * StateConsts.MAX_NUM_ENEMY_BULLETS_PER_COLUMN,
            shutter_small_distance_to_bullets=[0] * StateConsts.PRACTICAL_MAX_NUM_ENEMY_BULLETS_PER_SIDE,
            shutter_bullet_threat_binary=[0] * StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE,
            shutter_frames_until_collision=[float(InitialStateConsts.FRAMES_UNTIL_BULLET_COLLISION)] * StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE,
        )
        self.score_state = ScoreState(
            scores={'Human': 0, 'Shutter': 0, 'Nao': 0, 'NaoForHuman': 0, 'NaoForShutter': 0},
        )
        self.time_state = TimeState(
            frame=0,
            stamp=time.time()
        )
        self.reward_state = RewardState(
            sum_reward=0
        )
        self.history = History(
            average_shot_frequency_frames={'Human': None, 'Shutter': None, 'Nao': None},
            shot_frame_history={'Human': [], 'Shutter': [], 'Nao': []},
            last_shot_frame={'Human': float('-inf'), 'Shutter': float('-inf'), 'Nao': float('-inf')},
            last_hit_frame={'Human': float('-inf'), 'Shutter': float('-inf'), 'Nao': float('-inf')},
            last_enemy_shot_frame=0,
            support_frame_count={player: 0 for player in Players}
        )

        assert self.equal(self, ignore_missing_fields=False)

    def validate(self):
        self.players_state.validate()
        # self.enemy_state.validate()
        # self.bullet_state.validate()
        # self.score_state.validate()
        # self.time_state.validate()
        # self.reward_state.validate()
        # self.history.validate()
        #assert len(self.enemies_x) == StateConsts.NUM_ENEMIES
        #assert len(self.enemies_y) == StateConsts.NUM_ENEMIES
        # TODO:  more checks

    def equal(self, other, ignore_missing_fields=False, verbose=False):
        if not isinstance(other, SpaceInvadersState):
            raise ValueError(f"Cannot compare SpaceInvadersState with {type(other)}")

        if not self.players_state.equal(other.players_state, ignore_missing_fields=ignore_missing_fields):
            if verbose: print("Inequality found in players_state")
            return False
        if not self.enemy_state.equal(other.enemy_state, ignore_missing_fields=ignore_missing_fields):
            if verbose: print("Inequality found in enemy_state")
            return False
        if not self.bullet_state.equal(other.bullet_state, ignore_missing_fields=ignore_missing_fields):
            if verbose: print("Inequality found in bullet_state")
            return False
        if not self.score_state.equal(other.score_state, ignore_missing_fields=ignore_missing_fields):
            if verbose: print("Inequality found in score_state")
            return False
        if not self.time_state.equal(other.time_state, ignore_missing_fields=ignore_missing_fields):
            if verbose: print("Inequality found in time_state")
            return False
        if not self.reward_state.equal(other.reward_state, ignore_missing_fields=ignore_missing_fields):
            if verbose: print("Inequality found in reward_state")
            return False
        if not self.history.equal(other.history, ignore_missing_fields=ignore_missing_fields):
            if verbose: print("Inequality found in history")
            return False
        
        return True

    # TODO: add argument whether to return state dict as deep copy, true by default

    def to_dict(self):
        return copy.deepcopy({
            'players_state': self.players_state.to_dict(),
            'enemy_state': self.enemy_state.to_dict(),
            'bullet_state': self.bullet_state.to_dict(),
            'score_state': self.score_state.to_dict(),
            'time_state': self.time_state.to_dict(),
            'reward_state': self.reward_state.to_dict(),
            'history': self.history.to_dict(),
        })

    def _from_dict(self, state_dict, legacy_file_structure=False, require_all_attributes=False):
        if require_all_attributes:
            raise NotImplementedError("require_all_attributes is not implemented yet")

        self.players_state = AllPlayersState.from_dict(state_dict['players_state'], legacy_file_structure=legacy_file_structure)
        self.enemy_state = EnemyState.from_dict(state_dict['enemy_state'], legacy_file_structure=legacy_file_structure)
        self.bullet_state = BulletState.from_dict(state_dict['bullet_state'], legacy_file_structure=legacy_file_structure)
        self.score_state = ScoreState.from_dict(state_dict['score_state'], legacy_file_structure=legacy_file_structure)
        self.time_state = TimeState.from_dict(state_dict['time_state'], legacy_file_structure=legacy_file_structure)
        self.reward_state = RewardState.from_dict(state_dict['reward_state'], legacy_file_structure=legacy_file_structure)
        self.history = History.from_dict(state_dict['history'], legacy_file_structure=legacy_file_structure)

    @classmethod
    def from_dict(cls, state_dict, legacy_file_structure=False, require_all_attributes=False):
        self = cls(init_all_values=False)
        self._from_dict(state_dict, legacy_file_structure, require_all_attributes)
        return self

    def get_observation(self, config: 'SpaceInvadersConfig' = None, minimal_observation=False):
        """
        Capture the state of the game.
        """
        if config is None:
            config = self.config
        max_num_frames = config.game_duration_frames

        if minimal_observation:
            # Current frame to measure time in the game
            observation = [self.time_state.frame/max_num_frames]

            # Time since last shot for each agent
            for agent in Players.values():
                if self.history.last_shot_frame[agent] == float('-inf'):
                    last_shot_frame = ObservationConsts.NO_LAST_SHOT
                else:
                    last_shot_frame = self.history.last_shot_frame[agent]/max_num_frames
                observation.append(last_shot_frame)

            # TODO: can shoot booleans directly?

            # Active enemies
            #observation += [float(val) for val in self.enemy_state.enemies_active] # this is every enemy
            human_start_column_x = InitialStateConsts.HUMAN_ENEMIES_X[-2]
            active_enemies_start_column = []
            for (x, y), active in zip(zip(InitialStateConsts.ENEMIES_X, InitialStateConsts.ENEMIES_Y), self.enemy_state.enemies_active):
                if x == human_start_column_x:
                    active_enemies_start_column.append(float(active))
            observation += active_enemies_start_column

            # check if the observation is within the expected range
            observation = np.asarray(observation, dtype=ObservationConsts.DATA_TYPE)
            if np.any((observation < ObservationConsts.MIN_FEATURE_VALUE) | (observation > ObservationConsts.MAX_FEATURE_VALUE)):
                # Find indices where observation is less than -1 or greater than 1
                out_of_bounds_indices = np.where((observation < -1) | (observation > 1))[0]
                print(f"Observation indices out of bounds: {out_of_bounds_indices}")
                raise ValueError(f"Observation out of bounds: {observation}")
            
            return observation

        # Position of the human ship: center, left edge, and right edge.
        observation = [self.players_state.human_position_x]
        observation += [self.players_state.human_position_x - (DynamicsConsts.SHIP_WIDTH / 2)]  # Left edge
        observation += [self.players_state.human_position_x + (DynamicsConsts.SHIP_WIDTH / 2)]  # Right edge
        # Position of the shutter ship: center, left edge, and right edge.
        observation += [self.players_state.shutter_position_x]
        observation += [self.players_state.shutter_position_x - (DynamicsConsts.SHIP_WIDTH / 2)]  # Left edge
        observation += [self.players_state.shutter_position_x + (DynamicsConsts.SHIP_WIDTH / 2)]  # Right edge
        # Nao's position
        observation += [self.players_state.nao_position_x]

        # Player active statuses
        observation += [int(self.players_state.active[agent.value]) for agent in Players]

        #Enemy Positions
        observation += [float(val) for val in self.enemy_state.enemies_active]

        #Player bullets
        player_bullets_obs = self.bullet_state.player_bullets.copy()
        for bullet in player_bullets_obs:
            shooter = bullet[2]
            if shooter is None:
                shooter_index = ObservationConsts.NO_SHOOTER
            else:
                shooter_index = Players(shooter).index()
            bullet[2] = shooter_index            
            
        observation += list(player_bullets_obs.flatten(order='C'))

        #Enemy bullets
        flattened_enemy_bullets = self.bullet_state.bullet_y_positions.flatten(order='F')#Enemy bullets sorted by column
        observation += list(flattened_enemy_bullets)
        # Human distance to bullets
        observation += self.bullet_state.small_distance_to_bullets
        # Shutter distance to bullets
        observation += self.bullet_state.shutter_small_distance_to_bullets

        # Time in the game, by frame
        observation += [(self.time_state.frame/max_num_frames)]

        # Bullet threats and time-to-collision for both Human and Shutter
        observation += list(np.array(self.bullet_state.frames_until_collision)/max_num_frames)
        observation += list(self.bullet_state.bullet_threat_binary)
        observation += list(np.array(self.bullet_state.shutter_frames_until_collision)/max_num_frames)
        observation += list(self.bullet_state.shutter_bullet_threat_binary)
        
        #scores- normalized
        # TODO: what's 3500
        #observation += list([value/3500 for key, value in list(self.score_state.scores.items())[:2]])
        observation.append(self.score_state.scores[Players.HUMAN.value]/3500)
        observation.append(self.score_state.scores[Players.SHUTTER.value]/3500)
        observation.append(self.score_state.scores[Players.NAO.value]/3500)
        observation.append(self.score_state.scores['NaoForHuman']/3500)
        observation.append(self.score_state.scores['NaoForShutter']/3500)
        # check that more reliable way to get human/shutter scores in order works
        #assert observation[-2:] == list([value/3500 for key, value in list(self.score_state.scores.items())[:2]])

        # time since last shot for each agent
        # TODO: migrate all time variables to frames units
        #for agent in Players.values():
        #    observation.append((self.history.last_shot_time[agent])/1000)
        for agent in Players.values():
            if self.history.last_shot_frame[agent] == float('-inf'):
                last_shot_frame = ObservationConsts.NO_LAST_SHOT
            else:
                last_shot_frame = self.history.last_shot_frame[agent]/max_num_frames
            observation.append(last_shot_frame)

        # check if the observation is within the expected range
        observation = np.asarray(observation, dtype=ObservationConsts.DATA_TYPE)
        if np.any((observation < ObservationConsts.MIN_FEATURE_VALUE) | (observation > ObservationConsts.MAX_FEATURE_VALUE)):
            # Find indices where observation is less than -1 or greater than 1
            out_of_bounds_indices = np.where((observation < -1) | (observation > 1))[0]
            print(f"Observation indices out of bounds: {out_of_bounds_indices}")
            raise ValueError(f"Observation out of bounds: {observation}")
        
        return observation

    @staticmethod
    def from_observation(observation, config: 'SpaceInvadersConfig', default_init_all_values=False, minimal_observation=False):
        state = SpaceInvadersState(init_all_values=default_init_all_values)
        max_num_frames = config.game_duration_frames
        observation = copy.deepcopy(observation)

        if minimal_observation:
            FRAME_START = 0
            SHOT_FRAME_START = FRAME_START + 1
            ENEMIES_ACTIVE_START = SHOT_FRAME_START + 3
            FINAL_LEN = ENEMIES_ACTIVE_START + StateConsts.NUM_ENEMIES / StateConsts.NUM_ENEMY_COLUMNS

            state.time_state.frame = round(observation[FRAME_START]*max_num_frames)

            state.history.last_shot_frame = {}
            for i, agent in enumerate(Players.values()):
                last_shot_frame_fraction = observation[SHOT_FRAME_START+i]
                if last_shot_frame_fraction == ObservationConsts.NO_LAST_SHOT:
                    last_shot_frame = float('-inf')
                else:
                    last_shot_frame = round(last_shot_frame_fraction*max_num_frames)
                state.history.last_shot_frame[agent] = last_shot_frame

            
            state.enemy_state.enemies_active = [True] * StateConsts.NUM_ENEMIES
            human_start_column_x = InitialStateConsts.HUMAN_ENEMIES_X[-2]
            j_in_column = 0
            for i, (x, y) in enumerate(zip(InitialStateConsts.ENEMIES_X, InitialStateConsts.ENEMIES_Y)):
                if x == human_start_column_x:
                    state.enemy_state.enemies_active[i] = bool(observation[ENEMIES_ACTIVE_START+j_in_column])
                    j_in_column += 1

            state.players_state.active = {agent: True for agent in Players.values()}
            
            assert len(observation) == FINAL_LEN, f"Unexpected observation length: {len(observation)}, expected {FINAL_LEN}"

            return state

        # NOTE: index comments are out of date
        HUMAN_SHIP_START = 0
        SHUTTER_SHIP_START = HUMAN_SHIP_START + 3
        NAO_SHIP_START = SHUTTER_SHIP_START + 3
        PLAYERS_ACTIVE_START = NAO_SHIP_START + 1
        ENEMIES_ACTIVE_START  = PLAYERS_ACTIVE_START + 3
        PLAYER_BULLETS_START = ENEMIES_ACTIVE_START + StateConsts.NUM_ENEMIES
        ENEMY_BULLETS_START = PLAYER_BULLETS_START + StateConsts.MAX_NUM_PLAYER_BULLETS*ObservationConsts.NUM_PLAYER_BULLET_FEATURES
        HUMAN_BULLET_DISTANCE_START = ENEMY_BULLETS_START + StateConsts.NUM_ENEMY_COLUMNS*StateConsts.MAX_NUM_ENEMY_BULLETS_PER_COLUMN
        SHUTTER_BULLET_DISTANCE_START = HUMAN_BULLET_DISTANCE_START + StateConsts.PRACTICAL_MAX_NUM_ENEMY_BULLETS_PER_SIDE
        FRAME_START = SHUTTER_BULLET_DISTANCE_START + StateConsts.PRACTICAL_MAX_NUM_ENEMY_BULLETS_PER_SIDE
        HUMAN_TIME_TO_COLLISION_START = FRAME_START + 1
        HUMAN_BULLET_THREAT_START = HUMAN_TIME_TO_COLLISION_START + StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE
        SHUTTER_TIME_TO_COLLISION_START = HUMAN_BULLET_THREAT_START + StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE
        SHUTTER_BULLET_THREAT_START = SHUTTER_TIME_TO_COLLISION_START + StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE
        SCORES_START = SHUTTER_BULLET_THREAT_START + StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE
        SHOT_FRAME_START = SCORES_START + 5

        state.players_state.human_position_x = observation[HUMAN_SHIP_START]
        state.players_state.shutter_position_x = observation[SHUTTER_SHIP_START]
        state.players_state.nao_position_x = observation[NAO_SHIP_START]

        state.players_state.active = {agent: bool(observation[PLAYERS_ACTIVE_START+i]) for i, agent in enumerate(Players.values())}

        state.enemy_state.enemies_active = list(observation[ENEMIES_ACTIVE_START:ENEMIES_ACTIVE_START+StateConsts.NUM_ENEMIES].astype(bool))
        player_bullets_flat = np.copy(observation[PLAYER_BULLETS_START:PLAYER_BULLETS_START+StateConsts.MAX_NUM_PLAYER_BULLETS*ObservationConsts.NUM_PLAYER_BULLET_FEATURES]).astype(object)
        player_bullets = player_bullets_flat.reshape((StateConsts.MAX_NUM_PLAYER_BULLETS, ObservationConsts.NUM_PLAYER_BULLET_FEATURES), order='C')
        for bullet in player_bullets:
            shooter_index = int(bullet[2])
            if shooter_index == ObservationConsts.NO_SHOOTER:
                bullet[2] = None
            else:
                bullet[2] = Players.from_index(shooter_index).value
        state.bullet_state.player_bullets = player_bullets

        enemy_bullets_flattened = observation[ENEMY_BULLETS_START:ENEMY_BULLETS_START+StateConsts.NUM_ENEMY_COLUMNS*StateConsts.MAX_NUM_ENEMY_BULLETS_PER_COLUMN]
        state.bullet_state.bullet_y_positions = enemy_bullets_flattened.reshape((StateConsts.NUM_ENEMY_COLUMNS, StateConsts.MAX_NUM_ENEMY_BULLETS_PER_COLUMN), order='F')

        state.bullet_state.small_distance_to_bullets = list(observation[HUMAN_BULLET_DISTANCE_START:HUMAN_BULLET_DISTANCE_START+StateConsts.PRACTICAL_MAX_NUM_ENEMY_BULLETS_PER_SIDE])
        state.bullet_state.shutter_small_distance_to_bullets = list(observation[SHUTTER_BULLET_DISTANCE_START:SHUTTER_BULLET_DISTANCE_START+StateConsts.PRACTICAL_MAX_NUM_ENEMY_BULLETS_PER_SIDE])

        state.time_state.frame = round(observation[FRAME_START]*max_num_frames)

        state.bullet_state.frames_until_collision = list(observation[HUMAN_TIME_TO_COLLISION_START:HUMAN_TIME_TO_COLLISION_START+StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE]*max_num_frames)
        state.bullet_state.bullet_threat_binary = list(observation[HUMAN_BULLET_THREAT_START:HUMAN_BULLET_THREAT_START+StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE])
        state.bullet_state.shutter_frames_until_collision = list(observation[SHUTTER_TIME_TO_COLLISION_START:SHUTTER_TIME_TO_COLLISION_START+StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE]*max_num_frames)
        state.bullet_state.shutter_bullet_threat_binary = list(observation[SHUTTER_BULLET_THREAT_START:SHUTTER_BULLET_THREAT_START+StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE])

        # NOTE: observation doesn't include Nao's score
        state.score_state.scores = {"Human": observation[SCORES_START]*3500, "Shutter": observation[SCORES_START+1]*3500, "Nao": observation[SCORES_START+2]*3500,
                                    "NaoForHuman": observation[SCORES_START+3]*3500, "NaoForShutter": observation[SCORES_START+4]*3500}

        #state.history.last_shot_time = {agent: observation[SHOT_TIME_START+i] for i, agent in enumerate(Players.values())}
        state.history.last_shot_frame = {agent: observation[SHOT_FRAME_START+i]*max_num_frames for i, agent in enumerate(Players.values())}
        state.history.last_shot_frame = {}
        for i, agent in enumerate(Players.values()):
            last_shot_frame_fraction = observation[SHOT_FRAME_START+i]
            if last_shot_frame_fraction == ObservationConsts.NO_LAST_SHOT:
                last_shot_frame = float('-inf')
            else:
                last_shot_frame = round(last_shot_frame_fraction*max_num_frames)
            state.history.last_shot_frame[agent] = last_shot_frame

        assert len(observation) == SHOT_FRAME_START+3, f"Unexpected observation length: {len(observation)}, expected {SHOT_FRAME_START+3}"
        
        ### Compute derived variables
        enemy_bullets, enemyb_free, enemyb_used = state._get_enemybullets_from_bulletypositions()
        state.bullet_state.enemy_bullets = enemy_bullets
        state.bullet_state.enemyb_free = enemyb_free
        state.bullet_state.enemyb_used = enemyb_used

        return state
    
    def _get_enemybullets_from_bulletypositions(self):
        enemy_bullets = np.array([[0.,0.]] * StateConsts.MAX_NUM_ENEMY_BULLETS)
        enemyb_free = list(range(StateConsts.MAX_NUM_ENEMY_BULLETS))
        enemyb_used = []
        bullet_idx = 0
        for col in range(StateConsts.NUM_ENEMY_COLUMNS):
            for slot in range(StateConsts.MAX_NUM_ENEMY_BULLETS_PER_COLUMN):
                if self.bullet_state.bullet_y_positions[col, slot] > 0:
                    col_pixel = DynamicsConsts.COL_IDX_TO_PIXEL_MAP[col]
                    bullet_x = col_pixel / RenderConsts.SCREEN_WIDTH
                    bullet_y = self.bullet_state.bullet_y_positions[col, slot]
                    
                    enemy_bullets[bullet_idx] = [bullet_x, bullet_y]
                    enemyb_free.remove(bullet_idx)
                    enemyb_used.append(bullet_idx)
                    bullet_idx += 1

        return enemy_bullets, enemyb_free, enemyb_used
    
    def get_active_enemies_by_side(self):
        active_enemies_left, active_enemies_right = [], []
        for (x, y), active in zip(zip(InitialStateConsts.ENEMIES_X, InitialStateConsts.ENEMIES_Y), self.enemy_state.enemies_active):
            if active:
                if x >= 0.5:
                    active_enemies_right.append((x,y))
                else:
                    active_enemies_left.append((x,y))
        return active_enemies_left, active_enemies_right
    
    def get_enemy_bullets_by_side(self):
        bullets_left, bullets_right = [], []
        for i in self.bullet_state.enemyb_used:
            bullet_screen_x = int(self.bullet_state.enemy_bullets[i][0] * RenderConsts.SCREEN_WIDTH)
            if bullet_screen_x < RenderConsts.SCREEN_WIDTH / 2:
                bullets_left.append(self.bullet_state.enemy_bullets[i])
            else:
                bullets_right.append(self.bullet_state.enemy_bullets[i])
        return bullets_left, bullets_right
    
    def get_nearest_enemy(self, player, side_to_search=None, excluded_enemies=[], return_diffs=False):

        if player == Players.HUMAN:
            player_position_x = self.players_state.human_position_x
            player_position_y = InitialStateConsts.HUMAN_POSITION_Y
            if side_to_search is None:
                side_to_search = "left"
        elif player == Players.SHUTTER:
            player_position_x = self.players_state.shutter_position_x
            player_position_y = InitialStateConsts.SHUTTER_POSITION_Y
            if side_to_search is None:
                side_to_search = "right"
        elif player == Players.NAO:
            player_position_x = self.players_state.nao_position_x
            player_position_y = InitialStateConsts.NAO_POSITION_Y
            if side_to_search is None:
                side_to_search = "both"
        else:
            raise ValueError(f"Invalid player: {player}")

        active_enemies_left, active_enemies_right = self.get_active_enemies_by_side()
        if side_to_search == "left":
            enemies_to_search = active_enemies_left
        elif side_to_search == "right":
            enemies_to_search = active_enemies_right
        elif side_to_search == "both":
            enemies_to_search = active_enemies_left + active_enemies_right
        else:
            raise ValueError(f"Invalid side_to_search: {side_to_search}")


        nearest_enemy = [0,0]
        nearest_x_diff = 1 #RenderConsts.SCREEN_WIDTH
        nearest_y_diff = DynamicsConsts.VERTICAL_BUFFER
        closest_x_diff =1 # RenderConsts.SCREEN_WIDTH

        #In the Js version they looked through each enemy and checked if it was active or not. We are just going to check active enmies. 
        #Find nearest enemy to Nao Ship

        # The nearest enemy is: closest in y direction
        for enemy in enemies_to_search:
            if enemy in excluded_enemies:
                pass
            else:
                check_distance_x = abs(enemy[0] - player_position_x)
                check_distance_y = abs(enemy[1] - player_position_y)
                if check_distance_y < nearest_y_diff:
                    nearest_enemy = enemy
                    nearest_x_diff = check_distance_x
                    nearest_y_diff = check_distance_y
                elif check_distance_y == nearest_y_diff and check_distance_x < nearest_x_diff:
                    nearest_enemy = enemy
                    nearest_x_diff = check_distance_x
                    nearest_y_diff = check_distance_y
                elif check_distance_y < (nearest_y_diff+(DynamicsConsts.SHIP_HEIGHT)) and check_distance_x < closest_x_diff :
                    closest_x_diff = check_distance_x
        
        if return_diffs:
            return nearest_enemy, (nearest_x_diff, nearest_y_diff, closest_x_diff)
        else:
            return nearest_enemy
        
    def can_shoot(self, agent: Players) -> bool:
        #~~TODO: Changing this so that all agents can shoot the same frequency based ~~
        assert agent in Players
        
        if not self.players_state.active[agent.value]:
            return False

        frames_since_last_shot = self.time_state.frame - self.history.last_shot_frame[agent.value]
        max_human_shoot_frequency_frames = DynamicsConsts.MAX_HUMAN_SHOOT_FREQUENCY_FRAMES
        can_shoot = frames_since_last_shot > max_human_shoot_frequency_frames
        return can_shoot 

        # if agent == Players.HUMAN:
        #     max_human_shoot_frequency_frames = DynamicsConsts.MAX_HUMAN_SHOOT_FREQUENCY_FRAMES
        #     can_shoot = frames_since_last_shot > max_human_shoot_frequency_frames

        #     return can_shoot
        
        # human_avg_frequency_frames = DynamicsConsts.FREQUENCY_BOUND_FRAMES
        # if self.history.average_shot_frequency_frames['Human']:
        #     human_avg_frequency_frames = min(DynamicsConsts.FREQUENCY_BOUND_FRAMES, self.history.average_shot_frequency_frames['Human'])
        
        # if agent == Players.SHUTTER:
        #     can_shoot = frames_since_last_shot > human_avg_frequency_frames * DynamicsConsts.SHUTTER_RELATIVE_SPEED
           
        # elif agent == Players.NAO:
        #     can_shoot = frames_since_last_shot > human_avg_frequency_frames * DynamicsConsts.NAO_RELATIVE_SPEED
        
        # return can_shoot

        
    def get_leading_score_player(self, threshold=None) -> Optional[Players]:
        human_score = self.score_state.scores[Players.HUMAN.value] + self.score_state.scores['NaoForHuman']
        shutter_score = self.score_state.scores[Players.SHUTTER.value] + self.score_state.scores['NaoForShutter']
        if threshold is None:
            if human_score != shutter_score:
                return Players.HUMAN if human_score > shutter_score else Players.SHUTTER
            else:
                return None
        else:
            if abs(human_score - shutter_score) >= threshold:
                return Players.HUMAN if human_score > shutter_score else Players.SHUTTER
            else:
                return None
            
    def get_player_independent_victory(self, player: Players, config: 'SpaceInvadersConfig', include_robot_contributions: bool=True) -> bool:
        if player == Players.NAO:
            raise ValueError(f"Nao's victory cannot be checked independently, use {Players.HUMAN} or {Players.SHUTTER} instead.")        

        if config.independent_victory_score_threshold is None:
            return False      

        player_score = self.get_player_score(player, include_robot_contributions=include_robot_contributions)
        return player_score >= config.independent_victory_score_threshold
    
    def get_player_score(self, player: Players, include_robot_contributions: bool=True) -> int:
        if player == Players.NAO:
            assert self.score_state.scores[Players.NAO.value] == 0, "Nao's score should not be directly accessed, use NaoForHuman or NaoForShutter instead."
            if include_robot_contributions:
                return self.score_state.scores['NaoForHuman'] + self.score_state.scores['NaoForShutter']
            else:
                raise ValueError("Nao's score cannot be accessed directly, use include_robot_contributions=True to get the contributions from NaoForHuman and NaoForShutter.")
        elif player in [Players.HUMAN, Players.SHUTTER]:
            if include_robot_contributions:
                return self.score_state.scores[player.value] + self.score_state.scores[f"NaoFor{player.value}"]
            else:
                return self.score_state.scores[player.value] 
            
    def get_cumulative_num_enemy_eliminations(self, player: Players) -> int:
        """
        Returns the cumulative number of enemy eliminations for the given player.
        The cumulative score is calculated based on the player's score in the score state.

        Args:
            player (Players): The player for whom to calculate the cumulative number of enemy eliminations.
        Returns:
            int: The cumulative number of enemy eliminations for the given player.
        """
        if player == Players.HUMAN:
            cumulative_score = self.score_state.scores[Players.HUMAN.value]
        elif player == Players.SHUTTER:
            cumulative_score = self.score_state.scores[Players.SHUTTER.value]
        elif player == Players.NAO:
            cumulative_score = self.score_state.scores["NaoForHuman"] + self.score_state.scores["NaoForShutter"]

        # TODO: if score is no longer a simple scale of number of enemies eliminated, then track eliminations in state.History
        num_enemy_eliminations = int(cumulative_score / ScoreConsts.BONUS_FOR_HITTING_ENEMY)
        return num_enemy_eliminations
    
    def is_terminal(self, config: 'SpaceInvadersConfig' = None) -> bool:
        """
        Check if the game is in a terminal state.
        A terminal state is defined as:
        - In competitive mode, when the game duration has ended.
        - In competitive tutorial mode, when the player has eliminated the required number of enemies.
        
        Args:
            config (SpaceInvadersConfig, optional): The configuration of the game. If None, uses the state config.
        Returns:
            bool: True if the game is in a terminal state, False otherwise.
        """
        if config is None:
            config = self.config
        max_num_frames = config.game_duration_frames

        if config.game_type == GameTypes.COMPETITIVE:
            # +1 because the frame is 0-indexed, so the last frame is max_num_frames - 1
            is_terminal_step = self.time_state.frame + 1 >= max_num_frames
            return is_terminal_step
        elif config.game_type == GameTypes.COMPETITIVE_TUTORIAL:
            # The tutorial ends when the player has eliminated 5 enemies
            assert len(config.interactive_players) == 1
            interactive_player = config.interactive_players[0]
            num_enemy_eliminations = self.get_cumulative_num_enemy_eliminations(interactive_player)
            return num_enemy_eliminations >= DynamicsConsts.TUTORIAL_NUM_ENEMIES_TO_ELIMINATE
        else:
            raise ValueError(f"Invalid game type: {config.game_type}. Valid values are: {list(GameTypes)}")

@dataclass(frozen=True)
class SpaceInvadersConfig(StrictDataclass):
    action_space: ActionSpaces
    verbose: bool
    reward_model: RewardTypes
    render_mode: str # note: this is str because gym expects render mode to be a str
    support_policy: NaoSupportPolicies
    rules_based_human_policy: bool
    minimal_complexity_env: bool
    use_recent_states_buffer: bool
    use_observations: bool
    interactive_mode: InteractiveModes
    interactive_players: List[Players]
    num_game_segments: int
    game_duration_frames: int
    independent_victory_score_threshold: int
    game_type: GameTypes
    human_weak_player_flag: bool
    shutter_weak_player_flag: bool

    def __post_init__(self):
        super().__post_init__()

        if self.action_space not in ActionSpaces:
            raise ValueError(f"Invalid action space: {self.action_space}. Valid values are: {list(ActionSpaces)}")
        if self.reward_model not in RewardTypes:
            raise ValueError(f"Invalid reward model: {self.reward_model}. Valid values are: {list(RewardTypes)}")
        if self.render_mode not in RenderModes.values():
            raise ValueError(f"Invalid render mode: {self.render_mode}. Valid values are: {list(RenderModes.values())}")
        if self.support_policy not in NaoSupportPolicies:
            raise ValueError(f"Invalid support policy: {self.support_policy}. Valid values are: {list(NaoSupportPolicies)}")
        if self.interactive_mode not in InteractiveModes:
            raise ValueError(f"Invalid interactive mode: {self.interactive_mode}. Valid values are: {list(InteractiveModes)}")
        if self.interactive_players:
            if self.interactive_mode == InteractiveModes.NONE:
                raise ValueError(f"Interactive players cannot be set when interactive_mode is {InteractiveModes.NONE}. Set interactive_players to an empty list if you want no interactive players.")
            if not all(player in Players for player in self.interactive_players):
                raise ValueError(f"Invalid interactive players: {self.interactive_players}. Valid values are: {list(Players)}")
        if self.num_game_segments <= 0:
            raise ValueError(f"num_game_segments must be a positive integer, got {self.num_game_segments}")
        if self.game_duration_frames <= 0:
            raise ValueError(f"game_duration_frames must be a positive integer, got {self.game_duration_frames}")
        if self.independent_victory_score_threshold is not None and self.independent_victory_score_threshold <= 0:
            raise ValueError(f"independent_victory_score_threshold must be a positive integer, got {self.independent_victory_score_threshold}")
        if self.minimal_complexity_env:
            if self.reward_model != RewardTypes.MINIMAL_COMPLEXITY:
                raise ValueError(f"SpaceInvadersConfig: Minimal complexity environment requires minimal complexity reward model. Use reward_model={RewardTypes.MINIMAL_COMPLEXITY}")
            if self.action_space != ActionSpaces.HUMAN_ONLY:
                raise ValueError(f"SpaceInvadersConfig: Minimal complexity environment requires human-only action space. Use action_space={ActionSpaces.HUMAN_ONLY}")
            if not self.rules_based_human_policy:
                raise ValueError(f"SpaceInvadersConfig: Minimal complexity environment requires rules-based human policy. Use rules_based_human_policy=True")
        
@dataclass(frozen=True)
class SpaceInvadersRenderObjects:
    ship_image: pygame.Surface
    ship_inactive_image: pygame.Surface
    shutter_image: pygame.Surface
    shutter_inactive_image: pygame.Surface
    nao_image: pygame.Surface
    nao_inactive_image: pygame.Surface
    enemy_image: pygame.Surface
    enemy_image2: pygame.Surface
    bullet_image: pygame.Surface
    enemy_bullet_image: pygame.Surface
    font: pygame.font.Font
    small_font: pygame.font.Font


from consts import RewardConsts

@dataclass
class WelfareStepInfo(StrictDataclass):

    num_enemy_eliminations: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    num_player_hits: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    num_player_shots: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    num_player_taking_score_lead: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    num_player_losing_score_lead: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    player_victory: Dict[Players, bool] = field(default_factory=lambda: Players.default_dict(bool))

    @classmethod
    def from_step_info(cls, step_info: 'StepInfo', state: 'SpaceInvadersState', config: 'SpaceInvadersConfig') -> 'WelfareStepInfo':
        # self.num_enemy_eliminations = {reward/RewardConsts.ENEMY_ELIMINATION_REWARD for _player, reward in step_info.enemy_elimination_rewards.items()}
        # self.num_player_hits = {reward/RewardConsts.PLAYER_HIT_REWARD for _player, reward in step_info.player_hit_rewards.items()}
        # self.num_player_shots = {reward/RewardConsts.PLAYER_SHOOT_REWARD for _player, reward in step_info.player_shoot_rewards.items()}
        # self.num_player_taking_score_lead = # same pattern as previous, but only if reward is positive
        # self.num_player_losing_score_lead = # same pattern as previous, but only if reward is negative
        # self.player_victory = #boolean whether player won or not, based on step_info.player_victory_rewards[player] is greater than 0

        obj = cls()
        
        # Calculate the number of enemy eliminations
        obj.num_enemy_eliminations = {
            _player: reward / RewardConsts.ENEMY_ELIMINATION_REWARD
            for _player, reward in step_info.enemy_elimination_rewards.items()
        }

        # Calculate the number of player hits
        obj.num_player_hits = {
            _player: reward / RewardConsts.PLAYER_HIT_REWARD
            for _player, reward in step_info.player_hit_rewards.items()
        }

        # Calculate the number of player shots
        obj.num_player_shots = {
            _player: reward / RewardConsts.PLAYER_SHOOT_REWARD
            for _player, reward in step_info.player_shoot_rewards.items()
        }

        # Calculate the number of times a player took the score lead (which occurs when the reward is positive)
        obj.num_player_taking_score_lead = {
            _player: reward / RewardConsts.PLAYER_TAKING_SCORE_LEAD_REWARD
            for _player, reward in step_info.player_taking_score_lead_rewards.items()
            if reward > 0
        }

        # Calculate the number of times a player lost the score lead (which occurs when the reward is negative)
        obj.num_player_losing_score_lead = {
            _player: abs(reward) / RewardConsts.PLAYER_TAKING_SCORE_LEAD_REWARD
            for _player, reward in step_info.player_taking_score_lead_rewards.items()
            if reward < 0
        }

        # Determine whether each player achieved victory (boolean based on positive victory reward)
        # obj.player_victory = {
        #     _player: reward > 0
        #     for _player, reward in step_info.player_victory_rewards.items()
        # }
        #state.get_player_independent_victory()
        obj.player_victory = {
            _player: state.get_player_independent_victory(player=_player, config=config, include_robot_contributions=True) if _player != Players.NAO else False
            for _player in Players
        }

        return obj

@dataclass
class WelfareInfo(StrictDataclass):

    num_enemy_eliminations: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    num_player_hits: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    num_player_shots: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    num_player_taking_score_lead: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    num_player_losing_score_lead: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))
    player_victory: Dict[Players, bool] = field(default_factory=lambda: Players.default_dict(bool))

    starting_score: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    ending_score: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    support_frame_count: Dict[Players, float] = field(default_factory=lambda: Players.default_dict(float))

    @classmethod
    def from_all_steps(cls, all_welfare_step_info: List[WelfareStepInfo], initial_state: SpaceInvadersState, final_state: SpaceInvadersState) -> 'WelfareInfo':
        obj = cls()

        for welfare_step_info in all_welfare_step_info:
            assert isinstance(welfare_step_info, WelfareStepInfo)

            for player, num_enemy_eliminations in welfare_step_info.num_enemy_eliminations.items():
                obj.num_enemy_eliminations[player] += num_enemy_eliminations

            for player, num_player_hits in welfare_step_info.num_player_hits.items():
                obj.num_player_hits[player] += num_player_hits
            
            for player, num_player_shots in welfare_step_info.num_player_shots.items():
                obj.num_player_shots[player] += num_player_shots
            
            for player, num_player_taking_score_lead in welfare_step_info.num_player_taking_score_lead.items():
                obj.num_player_taking_score_lead[player] += num_player_taking_score_lead
            
            for player, num_player_losing_score_lead in welfare_step_info.num_player_losing_score_lead.items():
                obj.num_player_losing_score_lead[player] += num_player_losing_score_lead
            
            for player, player_victory in welfare_step_info.player_victory.items():
                obj.player_victory[player] = player_victory

        # Calculate support_frame_count for each player
        initial_support_frame_count = {
            Players.HUMAN: initial_state.history.support_frame_count[Players.HUMAN],
            Players.SHUTTER: initial_state.history.support_frame_count[Players.SHUTTER],
            Players.NAO: initial_state.time_state.frame - (initial_state.history.support_frame_count[Players.HUMAN] + initial_state.history.support_frame_count[Players.SHUTTER]),
        }
        final_support_frame_count = {
            Players.HUMAN: final_state.history.support_frame_count[Players.HUMAN],
            Players.SHUTTER: final_state.history.support_frame_count[Players.SHUTTER],
            Players.NAO: final_state.time_state.frame - (final_state.history.support_frame_count[Players.HUMAN] + final_state.history.support_frame_count[Players.SHUTTER]),
        }
        obj.support_frame_count = {
            player: final_support_frame_count[player] - initial_support_frame_count[player]
            for player in Players
        }
        # total_support_frames = sum(obj.support_frame_count.values())
        # total_scenario_frames = final_state.time_state.frame - initial_state.time_state.frame
        # if total_support_frames != total_scenario_frames:
        #     raise ValueError(f"Total support frames {total_support_frames} do not match total scenario frames {total_scenario_frames}. (start: {initial_state.time_state.frame}, end: {final_state.time_state.frame})")


        def remap_scores_dict(scores_dict):
            return {
                'Human+Nao': scores_dict[Players.HUMAN.value] + scores_dict['NaoForHuman'],
                'Shutter+Nao': scores_dict[Players.SHUTTER.value] + scores_dict['NaoForShutter'],
                'NaoForHuman': scores_dict['NaoForHuman'],
                'NaoForShutter': scores_dict['NaoForShutter'],
            }

        obj.ending_score = remap_scores_dict(final_state.score_state.scores)
        obj.starting_score = remap_scores_dict(initial_state.score_state.scores)
        # manually set nao for human and nao for shutter starting score to 0. Is this already the case by default?
        obj.starting_score['NaoForHuman'] = 0
        obj.starting_score['NaoForShutter'] = 0

        # assert that the starting score is less or equal than the ending score
        for scoring_player_combination in obj.starting_score.keys():
            if obj.starting_score[scoring_player_combination] > obj.ending_score[scoring_player_combination]:
                raise ValueError(f"Starting score {obj.starting_score[scoring_player_combination]} is greater than ending score {obj.ending_score[scoring_player_combination]} for player combination {scoring_player_combination}.")
        
        # # assert that support frames add up to the total frames in the scenario
        # total_support_frames = sum(obj.support_frame_count.values())
        # total_scenario_frames = final_state.time_state.frame - initial_state.time_state.frame
        # if total_support_frames != total_scenario_frames:
        #     raise ValueError(f"Total support frames {total_support_frames} do not match total scenario frames {total_scenario_frames}.")

        # assert that exactly one player was victorious
        # num_victorious_players = sum(1 for player, victory in obj.player_victory.items() if victory)
        # if num_victorious_players != 1:
        #     raise ValueError(f"Expected exactly one victorious player, but found {num_victorious_players}.")
        
        return obj
        
    def to_json(self):
        raise NotImplementedError("WelfareInfo.to_json() is out of date")
        # Convert the WelfareInfo object to a JSON-serializable dictionary
        welfare_dict = {
            'num_enemy_eliminations': {player.name: value for player, value in self.num_enemy_eliminations.items()},
            'num_player_hits': {player.name: value for player, value in self.num_player_hits.items()},
            'num_player_shots': {player.name: value for player, value in self.num_player_shots.items()},
            'num_player_taking_score_lead': {player.name: value for player, value in self.num_player_taking_score_lead.items()},
            'num_player_losing_score_lead': {player.name: value for player, value in self.num_player_losing_score_lead.items()},
            'player_victory': {player.name: value for player, value in self.player_victory.items()},
            'starting_score': {player_combination: value for player_combination, value in self.starting_score.items()},
            'ending_score': {player_combination: value for player_combination, value in self.ending_score.items()},
            'support_frame_count': {player.name: value for player, value in self.support_frame_count.items()},
        }

        # Also add vectorized features and names. These won't be used to reconstruct object
        welfare_dict["vectorized_features"] = {
            "feature_names": self.name_vectorized_features(),
            "vector": self.vectorize(human_focused=True).tolist(),
            "zipped": list(zip(self.name_vectorized_features(), self.vectorize(human_focused=True).tolist()))
        }
        return welfare_dict

    @classmethod
    def from_json(cls, welfare_dict) -> 'WelfareInfo':
        raise NotImplementedError("WelfareInfo.from_json() is out of date")
        # Convert the JSON data back to a WelfareInfo object
        welfare_info = cls(
            num_enemy_eliminations={Players[player]: value for player, value in welfare_dict['num_enemy_eliminations'].items()},
            num_player_hits={Players[player]: value for player, value in welfare_dict['num_player_hits'].items()},
            num_player_shots={Players[player]: value for player, value in welfare_dict['num_player_shots'].items()},
            num_player_taking_score_lead={Players[player]: value for player, value in welfare_dict['num_player_taking_score_lead'].items()},
            num_player_losing_score_lead={Players[player]: value for player, value in welfare_dict['num_player_losing_score_lead'].items()},
            player_victory={Players[player]: value for player, value in welfare_dict['player_victory'].items()},
            starting_score={player_combination: value for player_combination, value in welfare_dict['starting_score'].items()},
            ending_score={player_combination: value for player_combination, value in welfare_dict['ending_score'].items()},
            support_frame_count={Players[player]: value for player, value in welfare_dict['support_frame_count'].items()},
        )
        return welfare_info
    
    '''
    both players perspectives
    '''
    
    
    
    def name_vectorized_features(self, perspective="human_focused") -> List[str]:
        if perspective == "human_focused":
            event_count_features = [
                "num_enemy_eliminations",
                "num_hits_received",
                "num_shots_fired",
                "num_leads_taken",
                "num_leads_lost",
                "victory"
            ]

            # starting_score_human+nao, starting_score_shutter+nao, starting_score_nao_for_human, starting_score_nao_for_shutter
            # ending_score_human+nao, ending_score_shutter+nao, ending_score_nao_for_human, ending_score_nao_for_shutter
            score_features = [
                "starting_score_human+nao",
                "starting_score_shutter+nao",
                "starting_score_nao_for_human",
                "starting_score_nao_for_shutter",
                "ending_score_human+nao",
                "ending_score_shutter+nao",
                "ending_score_nao_for_human",
                "ending_score_nao_for_shutter"
            ]

            # frames_supporting_<player> for player in Players
            history_features = [
                f"frames_supporting_{player.name}" for player in Players
            ]
            return event_count_features + score_features + history_features
        elif perspective == "objective":
            features = []
            for player in [Players.HUMAN, Players.SHUTTER]:
                if player == Players.HUMAN:
                    player_name = "left"
                elif player == Players.SHUTTER:
                    player_name = "right"
                else:
                    raise ValueError(f"Invalid player: {player}")
                features += [
                    f"{player_name}_num_enemy_eliminations",
                    f"{player_name}_num_hits_received",
                    f"{player_name}_num_shots_fired",
                    f"{player_name}_victory"
                ]
                features += [
                    f"{player_name}_starting_score_with_robot",
                    f"{player_name}_starting_score_robot_contribution",
                    f"{player_name}_ending_score_with_robot",
                    f"{player_name}_ending_score_robot_contribution",
                ]
                features += [
                    f"{player_name}_frames_supporting_{player_name}",
                ]
            features += [
                "frames_supporting_neither",
            ]
            return features
        elif perspective in ["subjective_left", "subjective_right"]:
            # if perspective == "subjective_left":
            #     self_other_player_order = (Players.HUMAN, Players.SHUTTER)
            # elif perspective == "subjective_right":
            #     self_other_player_order = (Players.SHUTTER, Players.HUMAN)
            
            # TODO: share this code with objective perspective above, via helper function
            features = []
            # for player in self_other_player_order:
            #     if player == Players.HUMAN:
            #         player_name = "left"
            #     elif player == Players.SHUTTER:
            #         player_name = "right"
            #     else:
            #         raise ValueError(f"Invalid player: {player}")
            for player_name in ["self", "other"]:
                features += [
                    f"{player_name}_num_enemy_eliminations",
                    f"{player_name}_num_hits_received",
                    f"{player_name}_num_shots_fired",
                    f"{player_name}_victory"
                ]
                features += [
                    f"{player_name}_starting_score_with_robot",
                    f"{player_name}_starting_score_robot_contribution",
                    f"{player_name}_ending_score_with_robot",
                    f"{player_name}_ending_score_robot_contribution",
                ]
                features += [
                    f"{player_name}_frames_supporting_{player_name}",
                ]
            features += [
                "frames_supporting_neither",
            ]
            return features

        else:
            raise ValueError(f"Invalid perspective: {perspective}. Valid values are: 'human_focused', 'objective', 'subjective'")


    
    def vectorize(self, perspective="human_focused") -> np.ndarray:
        if perspective == "human_focused":
        
            num_enemy_eliminations = self.num_enemy_eliminations[Players.HUMAN]
            num_hits_received = self.num_player_hits[Players.HUMAN]
            num_shots_fired = self.num_player_shots[Players.HUMAN]
            num_leads_taken = self.num_player_taking_score_lead[Players.HUMAN]
            num_leads_lost = self.num_player_losing_score_lead[Players.HUMAN]
            victory = self.player_victory[Players.HUMAN]

            score_order = ['Human+Nao', 'Shutter+Nao', 'NaoForHuman', 'NaoForShutter']
            starting_score = [self.starting_score[score] for score in score_order]
            ending_score = [self.ending_score[score] for score in score_order]
            support_frame_counts = [self.support_frame_count[player] for player in Players]

            # Vectorize the welfare information
            vector = np.array([
                num_enemy_eliminations,
                num_hits_received,
                num_shots_fired,
                num_leads_taken,
                num_leads_lost,
                victory,
                *starting_score,
                *ending_score,
                *support_frame_counts,
            ], dtype=np.float32)
            return vector
        elif perspective == "objective":
            vector = []

            for player in [Players.HUMAN, Players.SHUTTER]:
                num_enemy_eliminations = self.num_enemy_eliminations[player]
                num_hits_received = self.num_player_hits[player]
                num_shots_fired = self.num_player_shots[player]
                victory = self.player_victory[player]

                starting_score_with_robot = self.starting_score[f"{player.name.capitalize()}+Nao"]
                starting_score_robot_contribution = self.starting_score[f"NaoFor{player.name.capitalize()}"]
                ending_score_with_robot = self.ending_score[f"{player.name.capitalize()}+Nao"]
                ending_score_robot_contribution = self.ending_score[f"NaoFor{player.name.capitalize()}"]
                
                support_frame_count = self.support_frame_count[player]

                vector += [
                    num_enemy_eliminations,
                    num_hits_received,
                    num_shots_fired,
                    victory,
                    starting_score_with_robot,
                    starting_score_robot_contribution,
                    ending_score_with_robot,
                    ending_score_robot_contribution,
                    support_frame_count,
                ]

            vector += self.support_frame_count[Players.NAO],
            return vector
        
        elif perspective in ["subjective_left", "subjective_right"]:
            # self_other_player_order = [
            #     (Players.HUMAN, Players.SHUTTER) if perspective == "subjective_left" 
            #     else (Players.SHUTTER, Players.HUMAN)
            # ]
            if perspective == "subjective_left":
                self_other_player_order = (Players.HUMAN, Players.SHUTTER)
            elif perspective == "subjective_right":
                self_other_player_order = (Players.SHUTTER, Players.HUMAN)
            
            # TODO: share this code with objective perspective above, via helper function
            vector = []
            for player in self_other_player_order:
                num_enemy_eliminations = self.num_enemy_eliminations[player]
                num_hits_received = self.num_player_hits[player]
                num_shots_fired = self.num_player_shots[player]
                victory = self.player_victory[player]

                starting_score_with_robot = self.starting_score[f"{player.name.capitalize()}+Nao"]
                starting_score_robot_contribution = self.starting_score[f"NaoFor{player.name.capitalize()}"]
                ending_score_with_robot = self.ending_score[f"{player.name.capitalize()}+Nao"]
                ending_score_robot_contribution = self.ending_score[f"NaoFor{player.name.capitalize()}"]
                
                support_frame_count = self.support_frame_count[player]

                vector += [
                    num_enemy_eliminations,
                    num_hits_received,
                    num_shots_fired,
                    victory,
                    starting_score_with_robot,
                    starting_score_robot_contribution,
                    ending_score_with_robot,
                    ending_score_robot_contribution,
                    support_frame_count,
                ]

            vector += self.support_frame_count[Players.NAO],
            return vector

        else:
            # TODO: use enumeration to specify the range of values
            raise ValueError(f"Invalid perspective: {perspective}. Valid values are: 'human_focused', 'objective', 'subjective'")
            
