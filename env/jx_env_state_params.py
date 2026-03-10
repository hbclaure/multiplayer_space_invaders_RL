import dataclasses
import typing
from typing import Any, Dict, Optional, Tuple, Union, Callable
import chex
from flax import struct
import jax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
import math
import os
# from consts import RenderConsts, StateConsts, ObservationConsts, ActionConsts, Players, Enemies 
# from consts import InitialStateConsts, DynamicsConsts, RewardConsts, ScoreConsts, PolicyConsts

# TODO: Ensure player bullet positions and bullet actives properly reflect the fact 
# max bullets per player is 4, 3, 3 respectively. Not 10 for all.
# TODO: use pytree_node=False w/ struct.field for any attributes that shouldn't be touched by Jax transformations

# Indexing convention: Indexing will always go left to right in the order of n_players, n_bullets, n_dims, side_bool, n_rows, n_cols

@struct.dataclass(slots=True)
class JxPlayerState(object):
    # Players State
    initial_player_positions: jax.Array # (n_players, n_dims)   --> (3, 2)
    player_positions: jax.Array         # (n_players, n_dims)   --> (3, 2)
    players_active: jax.Array           # (n_players,)          --> (3,)
    nao_supp_func_idx: int 

    @classmethod
    def reset(cls):
        initial_player_positions = jnp.array(
            [
                [0.25, 0.1], 
                [0.868, 0.1], 
                [0.5, 0.1]
            ]
        )
        
        player_positions = jnp.array(
            [
                [0.25, 0.1], 
                [0.868, 0.1], 
                [0.5, 0.1]
            ]
        )
        
        return cls(
            initial_player_positions=initial_player_positions, 
            player_positions=player_positions, 
            players_active=jnp.ones(shape=(3,), dtype=int), 
            nao_supp_func_idx=1
        )

@struct.dataclass(slots=True)
class JxEnemyState(object):
    # Enemy State
    initial_enemy_positions: jax.Array # (n_dims, side_bool, n_rows, n_cols) --> (2, 2, 5, 5)
    enemy_positions: jax.Array         # (n_dims, side_bool, n_rows, n_cols) --> (2, 2, 5, 5)
    enemies_active: jax.Array          # (side_bool, n_rows, n_cols)         --> (2, 5, 5)
    
    @classmethod
    def reset(cls):
        lhs_enemies_x = jnp.asarray([0.0314, 0.1062, 0.1814, 0.2564, 0.331]) # Enemies on human side only
        rhs_enemies_x = jnp.asarray([0.581, 0.655, 0.7314, 0.8064, 0.881]) + .088 # Enemies on shutter side only

        # lhs_y = jnp.array([0.543, 0.460, 0.376, 0.293, 0.210]).reshape(5, 1)
        # rhs_y = jnp.array([0.543, 0.460, 0.376, 0.293, 0.210]).reshape(5, 1)
        lhs_y = jnp.array([.210, .293, .376, .460, .543]).reshape(5, 1) + .2
        rhs_y = jnp.array([.210, .293, .376, .460, .543]).reshape(5, 1) + .2
    
        lhs_y_tiled = jnp.tile(lhs_y, (1, 5))
        rhs_y_tiled = jnp.tile(rhs_y, (1, 5))

        lhs_x_tiled = jnp.tile(lhs_enemies_x, (5, 1))
        rhs_x_tiled = jnp.tile(rhs_enemies_x, (5, 1))
        
        initial_enemy_positions = jnp.stack([jnp.stack([lhs_x_tiled, rhs_x_tiled]),
                                             jnp.stack([lhs_y_tiled, rhs_y_tiled])])

        enemy_positions = jnp.stack(arrays=[jnp.stack([lhs_x_tiled, rhs_x_tiled]), 
                                            jnp.stack([lhs_y_tiled, rhs_y_tiled])],
                                    dtype=float)
        
        enemies_active = jnp.ones(shape=(2, 5, 5), dtype=int)
        return cls(initial_enemy_positions=initial_enemy_positions, 
                   enemy_positions=enemy_positions, 
                   enemies_active=enemies_active)

@struct.dataclass(slots=True)
class JxBulletState(object):
    # Bullet State
    player_bullet_positions: jax.Array    # (n_players, n_bullets, n_dims) --> (3, 4, 2)
    player_bullet_actives: jax.Array      # (n_players, n_bullets)         --> (3, 4)
    max_bullets_allowed: jax.Array        # (n_players,)                   --> (3,) [4, 3, 3]
    
    enemy_bullet_positions: jax.Array     # (n_bullets, n_dims, side_bool, n_rows, n_cols) --> (2, 2, 2, 5, 5)
    enemy_bullet_actives: jax.Array       # (n_bullets, side_bool, n_rows, n_cols)        --> (2, 2, 5, 5)
    # TODO: enemy_bullet_actives: jax.Array # (n_bullets, side_bool, n_cols)        --> (2, 2, 5)
     
    frames_until_collision: jax.Array     # (n_players,) --> (3,)
    bullet_threat_binary: jax.Array       # (n_players, n_bullets, side_bool, n_rows, n_cols) --> (3, 2, 2, 5, 5)
    # TODO: bullet_threat_binary: jax.Array # (n_players, n_bullets, side_bool, n_cols)        --> (3, 2, 2, 5)

    #TODO: feel like this should be defined on each enemy instead of for all enemies
    enemy_shoot_counter: int

    @classmethod
    def reset(cls):
        player_bullet_positions = jnp.full(shape=(3, 4, 2), fill_value=jnp.inf, dtype=float)
        player_bullet_actives = jnp.zeros(shape=(3, 4), dtype=int)
        max_bullets_allowed = jnp.full(shape=3, fill_value=3, dtype=int)
        enemy_bullet_positions = jnp.full(shape=(2, 2, 2, 5, 5), fill_value=jnp.inf)
        enemy_bullet_actives = jnp.zeros(shape=(2, 2, 5, 5), dtype=int)
        frames_until_collision = jnp.full(shape=3, fill_value=0, dtype=float)
        bullet_threat_binary = jnp.zeros(shape=(3, 2, 2, 5, 5), dtype=int)
        shoot_counter = 0
        return cls(player_bullet_positions, player_bullet_actives, max_bullets_allowed, enemy_bullet_positions, enemy_bullet_actives, frames_until_collision, bullet_threat_binary, shoot_counter)
    
@struct.dataclass(slots=True)
class JxScoreState(object):
    # Score State
    scores: jax.Array           # (n_players,) --> (5,)
    player_hit_count: jax.Array # (n_players,) --> (3,)

    @classmethod
    def reset(cls):
        scores = jnp.zeros(shape=5, dtype=int)
        player_hit_count = jnp.zeros(shape=3, dtype=int)
        return cls(scores, player_hit_count)

@struct.dataclass(slots=True)
class JxTimeState(object):
    # Time State
    frame_num: int # ~~TODO: probably needs to be Array~~

    @classmethod
    def reset(cls):
        frame_num = 0
        return cls(frame_num)

@struct.dataclass(slots=True)
class JxRewardState(object):
    # Reward State
    sum_reward: jax.Array             # (n_players,) --> (3,)
    running_average_reward: jax.Array # (n_players,) --> (3,)

    @classmethod
    def reset(cls):
        sum_reward = jnp.zeros(shape=3, dtype=int)
        running_average_reward = jnp.zeros(shape=3, dtype=float)
        return cls(sum_reward, running_average_reward)

@struct.dataclass(slots=True)
class JxHistory(object):
    # History 
    player_frames_since_death: jax.Array # (n_players,) --> (3,)
    enemy_frames_since_death: jax.Array  # (side_bool, n_rows, n_cols) --> (2, 5, 5)                        
    player_last_shot_frame: jax.Array    # (n_players,) --> (3,)
    # TODO: Define this on each enemy instead of globally. Or at least for each side.
    enemy_last_shot_frame: int    
    support_frame_count: jax.Array       # (n_players,) --> (3,)
    last_hit_frame: jax.Array            # (n_players,) --> (3,)

    @classmethod
    def reset(cls):
        player_frames_since_death = jnp.full(shape=3, fill_value=-1, dtype=int)
        enemy_frames_since_death = jnp.full(shape=(2, 5, 5), fill_value=-1, dtype=int)
        # Initializing with -1000 so players can shoot on first turn 
        player_last_shot_frame = jnp.full(shape=3, fill_value=-1000, dtype=int)
        enemy_last_shot_frame = -1000
        support_frame_count = jnp.zeros(shape=3, dtype=int)
        last_hit_frame = jnp.zeros(shape=3, dtype=int)
        return cls(player_frames_since_death=player_frames_since_death, 
                   enemy_frames_since_death=enemy_frames_since_death, 
                   player_last_shot_frame=player_last_shot_frame, 
                   enemy_last_shot_frame=enemy_last_shot_frame, 
                   support_frame_count=support_frame_count, 
                   last_hit_frame=last_hit_frame)

@struct.dataclass(slots=True)
class JxStepInfo(object):
    # Step Info Rewards 
    # ~~TODO: Finish this out ~~
    enemy_elimination_rewards: jax.Array        # (n_players,) --> (3,)
    player_hit_rewards: jax.Array               # (n_players,) --> (3,)
    player_shoot_rewards: jax.Array             # (n_players,) --> (3,)
    player_taking_score_lead_rewards: jax.Array # (n_players,) --> (3,)
    player_victory_rewards: jax.Array           # (n_players,) --> (3,)
    score_changes: jax.Array                    # (n_players,) --> (3,)
    players_hit: jax.Array                      # (n_players,) --> (3,)
    players_respawned: jax.Array                # (n_players,) --> (3,)
    actions_taken: jax.Array                    # (n_players,) --> (3,4)

    @classmethod
    def reset(cls):
        enemy_elimination_rewards = jnp.zeros(shape=3, dtype=int)
        player_hit_rewards = jnp.zeros(shape=3, dtype=int)
        player_shoot_rewards = jnp.zeros(shape=3, dtype=int)
        player_taking_score_lead_rewards = jnp.zeros(shape=3, dtype=int)
        player_victory_rewards = jnp.zeros(shape=3, dtype=int)
        score_changes = jnp.zeros(shape=3, dtype=int)
        players_hit = jnp.zeros(shape=3, dtype=int)
        players_respawned = jnp.zeros(shape=3, dtype=int)
        actions_taken = jnp.zeros(shape=(3, 4), dtype=int)
        return cls(enemy_elimination_rewards, player_hit_rewards, player_shoot_rewards, player_taking_score_lead_rewards, player_victory_rewards, score_changes, players_hit, players_respawned, actions_taken)

@struct.dataclass(slots=True)
class EnvState(environment.EnvState):
    # TODO: Obs
    player_state: JxPlayerState
    enemy_state: JxEnemyState
    bullet_state: JxBulletState
    score_state: JxScoreState
    time_state: JxTimeState
    reward_state: JxRewardState
    history: JxHistory
    step_info: JxStepInfo
    sides: jax.Array 
    player_idxs: jax.Array
    # obs: jax.Array
    # seed: int
    # key: jax.Array

    @classmethod
    def reset(cls):
        player_state = JxPlayerState.reset()
        enemy_state = JxEnemyState.reset()
        bullet_state = JxBulletState.reset()
        score_state = JxScoreState.reset()
        time_state = JxTimeState.reset()
        reward_state = JxRewardState.reset()
        history = JxHistory.reset()
        step_info = JxStepInfo.reset()
        # obs = jnp.zeros(shape=1)
        # seed = 1
        # key = jax.random.key(seed=seed)
        return cls(
            player_state=player_state,
            enemy_state=enemy_state,
            bullet_state=bullet_state,
            score_state=score_state,
            time_state=time_state,
            reward_state=reward_state,
            history=history,
            step_info=step_info,
            sides=jnp.array([0, 1, 0], dtype=int),
            player_idxs=jnp.array([0, 1, 2], dtype=int),
            # seed=seed,
            # key=key,
            time=0 # have to do this because gymnax gives time as default
        )

    # def get_obs(self):
    #     return self.obs

@struct.dataclass
class JxRenderConsts:
    screen_width: int = 800
    screen_height: int = 600

    bullet_color: Tuple = (0, 255, 0) # Green
    black_background: Tuple = (0, 0, 0) # Black
    time_text_color: Tuple = (255, 255, 255) # White
    score_text_color: Tuple = (0, 255, 63) # Green

    ship_width: int = 50
    ship_height: int = 48
    bullet_width: float = 5
    bullet_height: float = 15

    IMAGE_DIR: str = struct.field(pytree_node=False, default="assets/images") 
    HUMAN_SHIP_IMAGE: str = struct.field(pytree_node=False, default="ship.png")
    SHUTTER_SHIP_IMAGE: str = struct.field(pytree_node=False, default="ship2.png")
    NAO_SHIP_IMAGE: str = struct.field(pytree_node=False, default="ai_ship.png")
    NAO_POWERUP_IMAGE: str = struct.field(pytree_node=False, default="nao_wings.png")
    
    ENEMY_IMAGE1: str = struct.field(pytree_node=False, default="enemy1_1.png")
    ENEMY_IMAGE2: str = struct.field(pytree_node=False, default="enemy3_1.png")
    BULLET_IMAGE: str = struct.field(pytree_node=False, default="laser.png")
    ENEMY_BULLET_IMAGE: str = struct.field(pytree_node=False, default="enemylaser.png")

    FONT_DIR: str = struct.field(pytree_node=False, default="assets/fonts/custom")
    FONT_FILENAME: str = struct.field(pytree_node=False, default="joystix.ttf")
    # TODO: couldn't figure out the string literal thing with os, so this is temp
    FONT_PATH: str = struct.field(pytree_node=False, default="assets/fonts/custom/joystix.ttf")
    # You can replace None with 'path/to/your/font.ttf' to use a custom font
    FONT_SIZE: int = 25

    SCORE_BOX_PLAYER1_VERTICAL_OFFSET: int = 20
    SCORE_BOX_PLAYER2_VERTICAL_OFFSET: int = 50

@struct.dataclass
class JxRewardConsts(object):
    enemy_elimination_reward: int = 5 # reward for eliminating an enemy
    player_hit_reward: int = -100 # reward/penalty for players getting hit by an enemy
    player_shoot_reward: int = 1 # reward for player shooting
    player_taking_score_lead_reward: int = 50 # reward for player taking the lead
    victory_reward: int = 500 # reward for winning the game
    leading_score_threshold: int = 50

@struct.dataclass
class JxScoreConsts(object):
    bonus_for_hitting_enemy: int = 10 

@struct.dataclass
class JxDynamicsConsts(object):

    vertical_buffer: float = 440 / 600 # TODO: what's 440?
    hit_range: float = 35 / 800 # range for dodging bullets- normalized. TODO: what's 35?
    second_hit_range: float = 50 /800 # TODO: what's 50, ship width?
    shooting_range: float = 24 / 800 #normalized TODO: what's 24? Ship height/2?
    ship_height: float = 48 / 600
    ship_width: float = 50 / 800
    min_x: int = 0
    move_left_bounds: jax.Array = jnp.array([0.0002, .5, 0])
    move_right_bounds: jax.Array = jnp.array([.35, 1, 1])
    
    # Timing and speed
    nao_relative_speed: float = 0.7 # relatively how much shorter time between bullets is for Shutter and Nao
    shutter_relative_speed: int = 1#0.8
    _frequency_bound: int = 3000 # max time between shots
    _max_human_shoot_frequency: int = 400
    enemy_bullet_velocity: float = -0.00277
    human_bullet_velocity: float = 0.00972
    min_enemy_shoot_frequency: int = 50
    respawn_frequency: int = 300
    _player_respawn_frequency_ms: int = 5000
    fps: int = 60
    player_shoot_bullet_speed: float = 0.01

    # Thresholds
    max_bullet_distance_to_include: float = 0.9

    # Termination conditions
    max_num_frames_per_game: int = 7200 # 2 minutes

    min_player_shoot_frequency_frames: float = (400 / 1000) * 60 # minimum number of frames between shots
    frequency_bound_frames: float = (3000 / 1000) * 60
    player_respawn_frequency_frames: float = (5000 / 1000) * 60

@struct.dataclass
class JxPolicyConsts:
    score_difference_threshold: int = 50
    support_history_balance_threshold: float = 0.25
    # TODO: should have value 'shoot' but needs pytree_node = False so jit ignores it 
    policy_mode: str = struct.field(pytree_node=False, default='shooting')
    # policy_mode: str = 'shooting'

@struct.dataclass
class JxStateConsts:
    num_sides: int = 2
    num_enemies: int = 50
    num_enemy_columns: int = 10
    max_num_enemy_bullets_per_column: int = 2
    num_enemy_columns_per_side: int = math.ceil(num_enemy_columns / num_sides)
    practical_max_num_enemy_bullets_per_side: int = 5
    max_num_player_bullets: int = 10
    max_num_enemy_bullets: int = 10

@struct.dataclass
class EnvParams(environment.EnvParams):
    """Environment parameters for Space Invaders"""

    # State Constants 
    state_consts: JxStateConsts = JxStateConsts()
    reward_consts: JxRewardConsts = JxRewardConsts()
    score_consts: JxScoreConsts = JxScoreConsts()
    dynamics_consts: JxDynamicsConsts = JxDynamicsConsts()
    policy_consts: JxPolicyConsts = JxPolicyConsts()
    render_consts: JxRenderConsts = JxRenderConsts()

    # Misc. 
    human_idx: int = 0
    shutter_idx: int = 1
    nao_idx: int = 2
    nao_for_human_score_idx: int = 3
    nao_for_shutter_score_idx: int = 4
    x_idx: int = 0
    y_idx: int = 1 