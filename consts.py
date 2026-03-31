import numpy as np
import os
from enum import Enum
from math import ceil
from units_utils import seconds_to_frames
from scipy.stats import truncnorm

class RenderConsts:

    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    RENDER_UPSCALE_FACTOR = 1.5 # Shutter: 2.25
    BULLET_COLOR = (0, 255, 0) # Green
    BLACK_BACKGROUND = (0, 0, 0) # Black
    TIME_TEXT_COLOR = (255, 255, 255) # White
    SCORE_TEXT_COLOR = (0, 255, 63) # Green

    SHIP_WIDTH = 50
    SHIP_HEIGHT = 48
    BULLET_WIDTH = 5
    BULLET_HEIGHT = 15

    IMAGE_DIR = "assets/images"
    HUMAN_SHIP_IMAGE = "ship.png"
    SHUTTER_SHIP_IMAGE = "ship2.png"
    NAO_SHIP_IMAGE = "ai_ship.png"
    NAO_POWERUP_IMAGE = "nao_wings.png"
    
    ENEMY_IMAGE1 = "enemy1_1.png"
    ENEMY_IMAGE2 = "enemy3_1.png"
    BULLET_IMAGE = "laser.png"
    ENEMY_BULLET_IMAGE = "enemylaser.png"

    FONT_DIR = "assets/fonts/custom"
    FONT_FILENAME = "joystix.ttf"
    FONT_PATH = os.path.join(FONT_DIR, FONT_FILENAME)
    # You can replace None with 'path/to/your/font.ttf' to use a custom font
    FONT_SIZE = 25
    SMALL_FONT_SIZE = 12

    PROGRESS_BAR_START_X = 100#210  # Fixed horizontal position where the bars start
    TEXT_PADDING = 10 # Padding between the text and the start of the bars, and right edge of screen
    SCORE_BOX_HEIGHT = 30
    SCORE_BOX_PLAYER1_VERTICAL_OFFSET = 10#20
    SCORE_BOX_PLAYER2_VERTICAL_OFFSET = SCORE_BOX_PLAYER1_VERTICAL_OFFSET + SCORE_BOX_HEIGHT#40#50

    # 2x magic number because victory threshold is about how much one can feasibly score in practice with robot help
    MAX_BAR_SIZE_VS_VICTORY_THRESHOLD_SCALE_FACTOR = 2
    # Theoretical upper limit = (10 points / hit) * (1 hit / 1 shot) * (1 shot / 50 frames) * (60 frames / second) * (N second game)
        # = 10 / 50 * 60 * N = 120 * (N seconds / game)

class ObservationConsts:

    OBS_SPACE_TYPE = "Box"

    MIN_FEATURE_VALUE = -1
    MAX_FEATURE_VALUE = 2
    # TODO: divide index of shooter by number of players to get back to -1 to 1 range

    NO_LAST_SHOT = -1
    NO_FRAMES_UNTIL_BULLET_COLLISION = -1
    NO_SHOOTER = -1

    DIM = 159
    SHAPE = (DIM,)
    MINIMAL_COMPLEXITY_DIM = 9
    MINIMAL_COMPLEXITY_SHAPE = (MINIMAL_COMPLEXITY_DIM,)

    DATA_TYPE = np.float32

    NUM_PLAYER_BULLET_FEATURES = 3

    RECENT_STATES_BUFFER_SIZE = 4


class ActionConsts:

    ACTION_SPACE_TYPE = "Discrete"

    DIM = 4

class RewardConsts:
    ENEMY_ELIMINATION_REWARD = 5 # reward for eliminating an enemy
    PLAYER_HIT_REWARD = -100 # reward/penalty for players getting hit by an enemy
    PLAYER_SHOOT_REWARD = 1 # reward for player shooting
    PLAYER_TAKING_SCORE_LEAD_REWARD = 50 # reward for player taking the lead
    VICTORY_REWARD = 500 # reward for winning the game

    LEADING_SCORE_THRESHOLD = 50


class PlayerPerformanceConsts:
    #threshold that has to be surpassed
    SINGLE_SHOOTING_THRESHOLD = .25 #.01 #np.random.normal(mu, sigma, num_samples)  #.01 For low skill, range goes from .01(about 330 ish for score) to .005 (290), .001(90 points)
    # .05 gives around 650 
    #.09 gives about 700
    #.15 gives about 750
    #.25 gives about 800
    HUMAN_SHOOTING_THRESHOLD = .25
    SHUTTER_SHOOTING_THRESHOLD = .25




    # mu = 0.01
    # sigma = 0.003
    # a = 0.001

    # a_std = (a - mu) / sigma

    # SHOOTING_THRESHOLD = truncnorm.rvs(a_std, np.inf, loc=mu, scale=sigma)

class ScoreConsts:
    BONUS_FOR_HITTING_ENEMY = 10
    #INDEPENDENT_VICTORY_THRESHOLD = None

class FairnessRewardConsts:
    TEAM_WEIGHT = 1.0
    OUTCOME_WEIGHT = 1.0
    TIME_WEIGHT = 1.0
    THRESHOLD_WEIGHT = 1.0
    EPSILON = 1e-8

# NOTE: constants prefixed with _ are configured in their units, but only used in code once converted to other units
class DynamicsConsts:
    # Spacing and sizes, many of these constants were set by the original version of the game
    VERTICAL_BUFFER = 440 / RenderConsts.SCREEN_HEIGHT # TODO: what's 440?
    HIT_RANGE = 35 / RenderConsts.SCREEN_WIDTH # range for dodging bullets- normalized. TODO: what's 35?
    SECOND_HIT_RANGE = 50 / RenderConsts.SCREEN_WIDTH # TODO: what's 50, ship width?
    SHOOTING_RANGE = 24 / RenderConsts.SCREEN_WIDTH #normalized TODO: what's 24? Ship height/2?
    SHIP_HEIGHT = RenderConsts.SHIP_HEIGHT / RenderConsts.SCREEN_HEIGHT
    SHIP_WIDTH = RenderConsts.SHIP_WIDTH / RenderConsts.SCREEN_WIDTH
    MIN_X = 0
    
    SAFE_CENTER_SPAWN_BUFFER_PER_SIDE = 0.1 # Space from center where players can safely spawn into
    HUMAN_LEFT_LIMIT = 0 + SHIP_WIDTH/2
    SHUTTER_LEFT_LIMIT = 0.5 + SAFE_CENTER_SPAWN_BUFFER_PER_SIDE + SHIP_WIDTH/2
    NAO_LEFT_LIMIT = 0 + SHIP_WIDTH/2
    HUMAN_RIGHT_LIMIT = 0.5 - SAFE_CENTER_SPAWN_BUFFER_PER_SIDE - SHIP_WIDTH/2
    SHUTTER_RIGHT_LIMIT = 1.0 - SHIP_WIDTH/2
    NAO_RIGHT_LIMIT = 1.0 - SHIP_WIDTH/2
    
    # Conversion between pixels and fractional coordinates
    #_group1_indexes = [25, 84, 145, 205, 264]  
    #_group2_indexes = [464, 524, 585, 645, 704]
    #_group1_indexes? = sorted(set(int(x*RenderConsts.SCREEN_WIDTH) for x in InitialStateConsts.ENEMIES_X))
    #PIXEL_TO_COL_IDX_MAP = {col: idx for idx, col in enumerate(_group1_indexes + _group2_indexes)}
    #COL_IDX_TO_PIXEL_MAP = {idx: px for px, idx in PIXEL_TO_COL_IDX_MAP.items()}
    
    # Timing and speed
    NAO_RELATIVE_SPEED = 0.7 # relatively how much shorter time between bullets is for Shutter and Nao
    SHUTTER_RELATIVE_SPEED = 1#0.8
    _FREQUENCY_BOUND = 3000 # max time between shots
    _MAX_HUMAN_SHOOT_FREQUENCY = 800#400
    _SHUTTER_NAO_SHOOT_FREQUENCY = 400

    BULLET_VELOCITY = 0.001#0.0015#0.00277
    HUMAN_BULLET_VELOCITY = -0.00972
    SHOOT_FREQUENCY = 50#10#50
    RESPAWN_FREQUENCY = 300 # frames=5 seconds
    _PLAYER_RESPAWN_FREQUENCY_MS = 3000#5000
    FRAMES_PER_SECOND = 60
    MAX_NUM_BULLETS_PER_SIDE = 4

    # Thresholds
    MAX_BULLET_DISTANCE_TO_INCLUDE = 0.9

    # Termination conditions
    MAX_NUM_FRAMES_PER_GAME = 3600*3#7200 # 2 minutes
    TUTORIAL_NUM_ENEMIES_TO_ELIMINATE = 25

    # Enemy spawning structure
    ENEMY_WAIT_FOR_HALF_ELIMINATION_TO_RESPAWN = True
    if ENEMY_WAIT_FOR_HALF_ELIMINATION_TO_RESPAWN:
        # Enemies spawn only after half of them are eliminated
        ENEMY_RESPAWN_FREQUENCY = 60 # frames=1 second


    # Breaks in the game
    NUM_GAME_SEGMENTS = 3

    # Non-interactive simulation
    NONINTERACTIVE_WAIT_TIME_SECONDS = 2

    @classmethod
    def initialize_derived_constants(cls):
        cls.MAX_HUMAN_SHOOT_FREQUENCY_FRAMES = seconds_to_frames(cls._MAX_HUMAN_SHOOT_FREQUENCY / 1000, cls.FRAMES_PER_SECOND)
        cls.MAX_SHUTTER_NAO_SHOOT_FREQUENCY_FRAMES = seconds_to_frames(cls._SHUTTER_NAO_SHOOT_FREQUENCY / 1000, cls.FRAMES_PER_SECOND)

        cls.FREQUENCY_BOUND_FRAMES = seconds_to_frames(cls._FREQUENCY_BOUND / 1000, cls.FRAMES_PER_SECOND)
        cls.PLAYER_RESPAWN_FREQUENCY_FRAMES = seconds_to_frames(cls._PLAYER_RESPAWN_FREQUENCY_MS / 1000, cls.FRAMES_PER_SECOND)

class PolicyConsts:
    SCORE_DIFFERENCE_THRESHOLD = 50
    SUPPORT_HISTORY_BALANCE_THRESHOLD = 0.25
    POLICY_MODE = 'shooting'
    NAO_INVINCIBLE = False
    _EQUAL_SUPPORT_INTERVAL_SECONDS = 15 #20? # seconds between support switches for Nao in equal support even policy
    _MIN_SUPPORT_DURATION_SECONDS = 5 # minimum duration of support in seconds, applies to selected policies
    
    #BIASED_SUPPORT_RATIO = 0.9 # ratio of support time for the favored player in biased support policies
    # Support ratio is implicitly _INTERVAL_TO_SUPPORT_UNFAVORED_PLAYER_SECONDS : _INTERVAL_TO_MAINTAIN_BIASED_RATIO_SECONDS
    _INTERVAL_TO_MAINTAIN_BIASED_RATIO_SECONDS = 60
    _INTERVAL_TO_SUPPORT_UNFAVORED_PLAYER_SECONDS = 10

    @classmethod
    def initialize_derived_constants(cls):
        cls.EQUAL_SUPPORT_INTERVAL_FRAMES = seconds_to_frames(cls._EQUAL_SUPPORT_INTERVAL_SECONDS, DynamicsConsts.FRAMES_PER_SECOND)
        cls.MIN_SUPPORT_DURATION_FRAMES = seconds_to_frames(cls._MIN_SUPPORT_DURATION_SECONDS, DynamicsConsts.FRAMES_PER_SECOND)
        cls.INTERVAL_TO_MAINTAIN_BIASED_RATIO_FRAMES = seconds_to_frames(cls._INTERVAL_TO_MAINTAIN_BIASED_RATIO_SECONDS, DynamicsConsts.FRAMES_PER_SECOND)
        cls.INTERVAL_TO_SUPPORT_UNFAVORED_PLAYER_FRAMES = seconds_to_frames(cls._INTERVAL_TO_SUPPORT_UNFAVORED_PLAYER_SECONDS, DynamicsConsts.FRAMES_PER_SECOND)

class EnhancedEnum(Enum):
    @classmethod
    def contains_value(cls, value):
        return value in {e.value for e in cls}
    
    @classmethod
    def values(cls):
        return [e.value for e in cls]
    
    @classmethod
    def from_index(cls, index):
        """Get the enum item from its index."""
        if index < 0 or index >= len(cls):
            raise ValueError(f"Index {index} is out of range for {cls.__name__}")
        return list(cls)[index]
    
    @classmethod
    def default_dict(cls, value_type, key_by_enum_values=False):
        keys = cls.values() if key_by_enum_values else list(cls)
        return {k: value_type() for k in keys}
    
    def index(self):
        return list(type(self)).index(self)


class Players(EnhancedEnum):
    HUMAN = "Human"
    SHUTTER = "Shutter"
    NAO = "Nao"

class Enemies(Enum):
    ENEMY1 = "Enemy1"
    ENEMY2 = "Enemy2"

class StateConsts:
    NUM_SIDES = 2

    NUM_ENEMIES = 50
    NUM_ENEMY_COLUMNS = 10
    MAX_NUM_ENEMY_BULLETS_PER_COLUMN = 2
    NUM_ENEMY_COLUMNS_PER_SIDE = ceil(NUM_ENEMY_COLUMNS / NUM_SIDES)
    PRACTICAL_MAX_NUM_ENEMY_BULLETS_PER_SIDE = 10#5

    MAX_NUM_PLAYER_BULLETS = 10
    MAX_NUM_ENEMY_BULLETS = 30#10

class InitialStateConsts:
    HUMAN_POSITION_X = 0.4
    HUMAN_POSITION_Y = 0.9
    SHUTTER_POSITION_X = 0.6
    SHUTTER_POSITION_Y = 0.9
    NAO_POSITION_X = 0.5
    NAO_POSITION_Y = 0.9
    # ENEMIES_X = [0.0314, 0.1062,0.1814,0.2564,0.331,0.0314, 0.1062,0.1814,0.2564,0.331,0.0314, 0.1062,0.1814,0.2564,0.331,0.0314, 0.1062,0.1814,0.2564,0.331,0.0314, 0.1062,0.1814,0.2564,0.331,0.581,0.655,0.7314,0.8064,0.881,0.581,0.655,0.7314,0.8064,0.881,0.581,0.655,0.7314,0.8064,0.881,0.581,0.655,0.7314,0.8064,0.881,0.581,0.655,0.7314,0.8064,0.881]
    # ENEMIES_Y = [0.543,0.543,0.543,0.543,0.543,0.460,0.460,0.460,0.460,0.460,0.376,0.376,0.376,0.376,0.376,0.293,0.293,0.293,0.293,0.293,0.210,0.210,0.210,0.210,0.210,0.543,0.543,0.543,0.543,0.543,0.460,0.460,0.460,0.460,0.460,0.376,0.376,0.376,0.376,0.376,0.293,0.293,0.293,0.293,0.293,0.210,0.210,0.210,0.210,0.210]
    # HUMAN_ENEMIES_X = [0.0314, 0.1062,0.1814,0.2564,0.331] #Enemies on human side only
    # SHUTTER_ENEMIES_X = [0.581,0.655,0.7314,0.8064,0.881] #Enemies on shutter side only
    FRAMES_UNTIL_BULLET_COLLISION = ObservationConsts.NO_FRAMES_UNTIL_BULLET_COLLISION

    @classmethod
    def compute_enemy_positions(cls):
        # Define boundaries (technically boundaries for center of the enemies)
        left_enemies_left_boundary = 0
        left_enemies_right_boundary = 0.5 - DynamicsConsts.SAFE_CENTER_SPAWN_BUFFER_PER_SIDE
        right_enemies_left_boundary = 0.5 + DynamicsConsts.SAFE_CENTER_SPAWN_BUFFER_PER_SIDE
        right_enemies_right_boundary = 1

        # Compute enemy positions
        cls.HUMAN_ENEMIES_X = cls._compute_enemy_x_positions(
            left_enemies_left_boundary,
            left_enemies_right_boundary,
            StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE
        )
        cls.SHUTTER_ENEMIES_X = cls._compute_enemy_x_positions(
            right_enemies_left_boundary,
            right_enemies_right_boundary,
            StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE
        )
        cls.ENEMIES_UNIQUE_Y = cls._compute_enemy_y_positions(
            highest_enemy_y_position=0.25,
            lowest_enemy_y_position=0.6,
            num_enemies_per_column=StateConsts.NUM_ENEMIES // StateConsts.NUM_ENEMY_COLUMNS
        )
        cls.ENEMIES_X, cls.ENEMIES_Y, cls.ENEMY_IDX_TO_COL_IDX, cls.ENEMY_X_TO_COL_IDX = cls._compute_full_enemy_positions(
            cls.HUMAN_ENEMIES_X,
            cls.SHUTTER_ENEMIES_X,
            cls.ENEMIES_UNIQUE_Y
        )

        assert len(cls.ENEMIES_X) == StateConsts.NUM_ENEMIES
        assert len(cls.ENEMIES_Y) == StateConsts.NUM_ENEMIES
        assert len(cls.HUMAN_ENEMIES_X) == StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE
        assert len(cls.SHUTTER_ENEMIES_X) == StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE

    @staticmethod
    def _compute_enemy_x_positions(left_boundary, right_boundary, num_columns):
        """Compute evenly spaced x positions for enemies."""
        segment_width = (right_boundary - left_boundary) / num_columns
        return [
            left_boundary + (column_index * segment_width) + segment_width / 2
            for column_index in range(num_columns)
        ]

    @staticmethod
    def _compute_enemy_y_positions(highest_enemy_y_position, lowest_enemy_y_position, num_enemies_per_column):
        """Compute evenly spaced unique y positions for enemies."""
        return [
            highest_enemy_y_position + i * (lowest_enemy_y_position - highest_enemy_y_position) / num_enemies_per_column
            for i in range(num_enemies_per_column)
        ]

    @staticmethod
    def _compute_full_enemy_positions(human_enemies_x, shutter_enemies_x, enemies_unique_y):
        """Compute full x and y positions for all enemies."""
        enemies_x, enemies_y = [], []
        num_enemies_per_column = len(enemies_unique_y)
        enemy_idx_to_col_idx = {}
        enemy_x_to_col_idx = {}

        # Add human-side enemies
        for col_idx, x in enumerate(human_enemies_x):
            for y in enemies_unique_y:
                enemies_x.append(x)
                enemies_y.append(y)
                enemy_idx_to_col_idx[len(enemies_x) - 1] = col_idx
            enemy_x_to_col_idx[x] = col_idx

        # Add shutter-side enemies
        for col_idx, x in enumerate(shutter_enemies_x):
            for y in enemies_unique_y:
                enemies_x.append(x)
                enemies_y.append(y)
                enemy_idx_to_col_idx[len(enemies_x) - 1] = col_idx + StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE
            enemy_x_to_col_idx[x] = col_idx + StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE

        return enemies_x, enemies_y, enemy_idx_to_col_idx, enemy_x_to_col_idx

# Compute enemy positions    
InitialStateConsts.compute_enemy_positions()
DynamicsConsts.initialize_derived_constants()
PolicyConsts.initialize_derived_constants()

class GameTypes(EnhancedEnum):
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"
    COMPETITIVE_TUTORIAL = "competitive_tutorial"

class RewardTypes(EnhancedEnum):
    FULL = "full"
    MINIMAL_COMPLEXITY = "minimalComplexity"
    NAO_FAIRNESS = "naoFairness"

class RenderModes(EnhancedEnum):
    RGB_ARRAY = "rgb_array"
    DISPLAY_WINDOW = "window"
    DISPLAY_FULLSCREEN = "fullscreen"

class NaoSupportPolicies(EnhancedEnum):
    EQUAL_SUPPORT = "equalSupport"
    HUMAN_ONLY = "humanOnly"
    SHUTTER_ONLY = "shutterOnly"
    TIMED_EQUAL_SUPPORT = "timedEqualSupport"
    PARTIAL_SUPPORT_HUMAN = "partialSupportHuman"
    EQUALIZE_SCORES = "equalizeScores"
    EQUALIZE_SUPPORT_HISTORY = "equalizeHistory"
    EQUALIZE_SUPPORT_HISTORY_EVEN = "equalizeHistoryEven"
    IDLE = "idleSupport"
    EQUALIZE_SCORES_NEUTRAL = 'equalizeScoresNeutral'
    BIASED_SUPPORT_LEFT = "biasedSupportLeft"
    BIASED_SUPPORT_RIGHT = "biasedSupportRight"

class HumanPolicies(EnhancedEnum):
    RULES_BASED = "rulesBased"
    OPTIMAL = "optimal"

class MinimalComplexityHumanPolicies(EnhancedEnum):
    SHOOT_WHEN_READY = "shootWhenReady"
    ALWAYS_SHOOT = "alwaysShoot"
    NOT_APPLICABLE = None

class ActionSpaces(EnhancedEnum):
    HUMAN_ONLY = "HumanOnly"
    SHUTTER_ONLY = "ShutterOnly"
    NAO_ONLY = "NaoOnly"
    JOINT_AGENT = "JointAgent"
    


class ValueFunctionType(EnhancedEnum):
    Q_FUNCTION = "QFunction"
    MC_ROLLOUT = "MCRollout"

class PlayerShootingAdjustment(EnhancedEnum):
    #Choose which player to adjust the shooting level of
    HUMAN= 'human'
    SHUTTER = 'shutter'
    BOTH = 'both'


def action_space_to_player(action_space):
    if action_space == ActionSpaces.HUMAN_ONLY:
        return Players.HUMAN
    elif action_space == ActionSpaces.SHUTTER_ONLY:
        return Players.SHUTTER
    elif action_space == ActionSpaces.NAO_ONLY:
        return Players.NAO
    elif action_space == ActionSpaces.JOINT_AGENT:
        raise NotImplementedError("Joint agent not supported")
    else:
        raise ValueError(f"Invalid action space: {action_space}")

# def sample():
#     if np.random.rand() < 0.5:
#         return np.random.normal(0.01, 0.003), "low_skill_player"
#     else:
#         return np.random.normal(0.15, 0.027), "high_skill_player" 

def sample_truncated_normal(mu, sigma, low=0.0, high=1.0):
    a = (low - mu) / sigma
    b = (high - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)

def sample():
    if np.random.rand() < 0.5:
        return sample_truncated_normal(0.01, 0.003), "low_skill_player"
    else:
        return sample_truncated_normal(0.15, 0.027), "high_skill_player"
    
class InteractiveModes(EnhancedEnum):
    NONE = "none"
    KEYBOARD = "keyboard"
    GAMEPAD = "gamepad"
