import random
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from agents.nao_policy import NaoPolicy
import copy
from consts import RenderConsts, StateConsts, ObservationConsts, ActionConsts, Players, Enemies, PlayerPerformanceConsts
from consts import InitialStateConsts, DynamicsConsts, RewardConsts, ScoreConsts, PolicyConsts, FairnessRewardConsts
from env_utils import load_ship_image, load_bullet_image, SpaceInvadersState, SpaceInvadersConfig, SpaceInvadersRenderObjects
from env_utils import index_to_boolean_policy, boolean_policy_to_index, tie_breaker_actions, frames_to_seconds, seconds_to_frames
from agents.policies import human_rules_based_policy_from_state
from consts import RewardTypes, NaoSupportPolicies, RenderModes, ActionSpaces, action_space_to_player, InteractiveModes, GameTypes
from env_utils import StepInfo
from agents.policies import SingleAgentAction
from agents.control_mapping import PyGameGamepad
from interactive_gameplay.utils import PygameEventWaiter, PyGameGamepad


from collections import deque
from typing import Deque

#Removed from original [nearest_enemy_nao]Need to find a way to include
        # nearest_enemy, (nearest_x_diff, nearest_y_diff, closest_x_diff) = (
        #     self.state.get_nearest_enemy(
        #         player=Players.SHUTTER,
        #         side_to_search="right",
        #         excluded_enemies= [],#[nearest_enemy_nao],
        #         return_diffs=True,
        #     )
        # )
        

FORCE_STEP_BY_FRAME_RATE = False
TEST_OBSERVATION_STATE_CYCLE = False
TEST_JSON_SAVE_CYCLE = False

DRAW_CENTER_AND_SPAWN_LINES = False

SPLIT_SCORE_BAR = True

# DONE: Osa: Adjust equalize_scores policy to account for joint scores, not just individual score (true for all policies)
# TODO: Osa: Add policy for equalizing scores based on individual component?
# TODO: Osa: Figure out why shutter pauses for a bit at start of the game
# DONE: Osa: Stop following behavior again from Nao (probably randomize left and right when it can)
        #    Added back supported nearest enemy for Nao policy
# DONE: Osa: Standardize can_shoot behavior for all agents 
# DONE: Osa: Change display to display total score as sum of nao and shutter/human component, instead of just the addition sign 
# DONE: Osa: Why does shutter policy use nearest enemy diff and closest x diff, but human isn't?
        #    Shutter now uses nearest enemy only as opposed to nearest enemy and closest x diff to match human
# DONE: Osa: Why is shutter using x_diff_prev, but human isn't? 
        #    x_diff_prev and abs(nearest_bullet[0] - state.players_state.human_position_x) should be the same
# DONE: Osa: Why is shutter using (200/300), but human uses (200 / RenderConsts.SCREEN_HEIGHT)
        #    Shutter now also uses RenderConsts.SCREEN_HEIGHT instead of 300
# TODO: minor stylistic
# DONE: add actions of agents in the environment to info
# TODO: restyle action_handler to be general to selected agent not being human
# TODO: rename bullet_y_positions to indicate enemy bullets, other bullet variables explicitly indicate player or enemy
# DONE: refer to constants directly instead  of copying them to instance variables
# DONE: bundle together all variables used to track the dynamic state into a State class and work with that for saving/loading/resetting
# TODO: share code between competitive_env.py and cooperative_env.py, put it in env_utils.py
# TODO: share code between nao policy, shutter policy, and human policy
# TODO: define variables or comments identifying remaining magic numbers (75, 55, 262, 3500, )
# TODO: make all times in milliseconds to avoid potential confusion
# TODO: manage pixel vs fractional variable names more consistently
# TODO: rename bullet velocity to bullet speed and sign it accordingly
# TODO: rename bullet velocity to enemy bullet velocity; in general more explicit about what's enemy vs player vars
# TODO: one constant for the game length in various units
# TODO: proper use of Players enumeration to generalize number and names of players
# TODO: rename shoot_counter to enemy_shoot_counter, plus other enemy vs player variables. active to player active.
# DONE: remove unused variables like start_time, enemy_bullets_x/y_rel, (enemy_)bullets_x/y
# DONE: remove both nao_increasing_player_score and support. Nao policy tracks support internally.
# DONE: remove shot_time_history_shutter: not used anywhere and redundant with shot_time_history
# DONE: only track sum reward and not running avg since running avg is always computed from sum reward? Running avg only used for printing
# DONE: add constants for 10 and 2 magic numbers in update_bullet_positions()
# DONE: remove spaceships_left/right/upper_edge: they are redundant with human_position_x/y. Only need a function to compute them locally based on human position from state
# TODO: convert players_state into dictionary of individual player state objects to share code/variables between them
# NOTE: last shot time and last enemy shot time are not using the same time units
# TODO: add str function to state class
# TODO: use index variables from from_observation() function in to_observation function to help keep consistent. Put both in env_utils.py
# TODO: share computation of derived variables like active enemies and bullets by left/right side: compute once and pass to all agent policies
# TODO: migrate shutter policy and nao policy to use state class
# TODO: collapse state field bullet_y_positions into enemy_bullets, and compute y positions shape as needed
# TODO: collapse average shot/frame frequency out of state variable and compute from shot time/frame history as needed
# TODO: replace player_position_y with InitialStateConsts.PLAYER_POSITION_Y
# TODO: rename left/right side to human/nao side, and define function to determine which side any coordinate/entity is on
# DONE: migrate _can_shoot to state class
# TODO: type hints, especially for state class
# TODO: nao policy, make support 0 or 1 into constants reflecting human vs shutter
# TODO: rename closest_x_diff to second_nearest_x_diff?
# TODO: last shot frame: use obsconsts value -1 instead of inf, and check for -1 in env logic
# TODO?: update frames_until_collision to usse obsconsts value -1 as feature with special handling instead of computing fraction as usual
# TODO: rename human respawn handler to player respawn handler
# TODO: migrate shutter policy out of env into policies, and compute nearest enemy nao live rather than as an argument, like human policy
# TODO: rename left,right,shoot actions in step() to nao_left, nao_right, nao_shoot
# TODO: reset/init function for each state subclass

# TODO: major functional:
# DONE: check if all features are  actually bounded by -1 and +1
# DONE: make sure self.spaceship_left_edge and right/upper edges update with self.human_position_x/y  (fixed: Houston)
# DONE: replace time.time() timing logic with frame-based logic
    # DONE: migrate last_shot_time to last_shot_frame
    # DONE: migrate update_shot_time_history() to use frames instead of time. Convert relevant consts to frames
    # DONE: all time-based thresholds are not consistent during dqn rollouts which are faster than 60 fps. Must be frame-based
# DONE: check on redundancy of bullet variables, e.g. self.enemy_bullets, self.enemy_bullets_x/y, self.enemyb_free/used
    # Seems like enemy_bullets and bullet_y_positions are in sync with each other but redundant. Only using one of them is less error prone
# TODO: in general, check for multiple variables that must track each other/update together
# TODO: in general, consolidate and remove redundant variables
# DONE: observation featurization: safer way to get human and shutter scores without assuming order of dictionary keys
# DONE: check whether nao_increasing_player_score is properly reset: wiped in get_obs() and set in check_collisions(). (fixed: Houston). Variable isn't used, so removed it
# TODO: check on nao/shutter/human ship positions: all all units consistent, pixels vs fractions?
# TODO: add checks on state variable values to ensure they are within expected bounds and consistent with each other
# DONE: check the directionality and sorting with small distance to bullets
# DONE: check on duplication of state variables inside Nao policy: support, nao_shoots. All policies should run from the same state. Functionally done, but stylistically not using State class yet
# DONE: is sum reward incremented twice in step()?
# DONE: all players immediately re-activate upon getting hit, and no score penalty. Should they respawn after some amount of time?
    # currently, all players respawn immediately, and are marked inactive after a hit
    # now respawn after 5 seconds
# TODO: should bullet-bullet collisions cancel each other out?

# NOTE: design decisions
# NOTE: average shot frequency is a rolling average that's iniitially arbitrarily large since current time/frame - init=0/-inf is ~inf. 
    # Avg frequency is throttled to frequency bound so this doesn't get out of hand
    # More accurate frequency until first gets pushed out after 5 shots would be to omit adding the first shot time to history, but may not be a big deal
# NOTE: nao and shutter policies can return multiple actions (move and shoot). Joint policy requires only one at a time.
    # Now, action_handler() has a tiebreaker to randomly split ties. Random okay? Or should we prioritize shooting over moving?
# NOTE: restrict learned policy actions? If policy says to shoot but that's not possible, we force a no-op. 
    # Seems okay to do that now that observation includes last shot timing to inform policy that a shot won't go through
# NOTE: enemy bullets are force-aligned to enemy columns, but player bullets are not
    # so enemy bullets are inherently grouped into slots corresponding to the number of enemy columns
    # but player bullets are fired from the exact spaceship position and slot indices are arbitrary
# NOTE: do any timers other than pygame need to reset upon restore_rendered_objects()? Don't think so
# NOTE: human shoot frequency const is 400, but human policy uses 500 to determine if shooting is possible. Now using const.py's 400 in shared call to can_shoot(). Okay? 
# NOTE: some variables were in init() but not reset(). Now all are always reset in init() and reset():
    # active_enemies_left_side, active_enemies_right_side: but cleared every human policy col. May have gotten lucky and avoided bug so far
    # average_shot_frequency: just a state to determine shutter shooting, but its rolling average that's never reset. Intentional?
    # shoot_counter: high frequency counter that cycles to determine when enemies can shoot. Not reset, but that just starts cycle partway through so no big deal
    # shot_time_history: not used directly, but used to determine average_shot_frequency
    # support: it flips cylcically every 1000 frames. Not reset, but no big deal to start each episode partway through cycle
    # clock: does pygame need clock to be reset after each episode ??? Now resetting as part of state object upon reset(). Is that okay?
    # start_time: isn't actually used for anything


# Order of operations

    # DONE: migrate human_policy_from_obs to use self.state parsed from observation vector, not env
    # DONE: update loading from observation to fill in remaining state variables human policy needs
    # DONE: migrate nearest enemy nao to helper function shared for each policy
    # DONE: migrate any state changes upon executing human policy to step() function. Policy should change nothing about the state
        # DONE: helper functions to update bullets or parse shooting availability should happen in step() function
        # DONE: step() should take given human action and enact it along with shutter/nao actions and passive changes each timestep
    # DONE: migrate shoot timing constraints to frame-based instead of time-based
        # DONE: human policy
        # DONE: shutter policy
        # DONE: nao policy
        # DONE: enemies
    
    # test rules-based policy's observation to state with state init to empty variables to make sure all
        # needed variables are parsed from observation or from constants

    # test human rules-based policy q learning and optimal policy learning

    # create shared helper functions for nao policy, compare output to existing version using copied state
    # propagate hitbox changes in helper functions to other agent policies
    
    # convert shutter and nao policies to use from observation wrapper
    # migrate all state changes in shutter and nao policies to the step function
    # test joint q learning

    # fix hitbox adjustment in nao_policy to integrate bukhosi's branch

    # enable state class to validate the state
    # check for known flagged potential bugs
    # add stylistic changes to prevent bugs


# Multiplayer
'''
Updates for later if needing gym env to reflect multiplayer dynamics:
    Update the action space to be a joint action space for all agents. Rollout loop will choose policy action for each player and for nao.
    Action space is discrete space reflecting the combination of actions by each player

For now, just leave the action space as a single player's actions? 
We want the recorded trajectories to record each player's actions, but the action space is only for the selected agent.
    StepInfo records the actions of every agent regardless of the action space for gym

First, get keyboard inputs working with one player

Then, set up environment to function with two players. Then get the rollout_trajectory() loop to work with keyboard inputs for both players
    Might be good enough to keep single agent action space where one player has their controller input externally passed to step() and other's is determined within step()

    If that gets too ugly, then update action space to be joint action space for all agents so that step() takes in joint action for every agent and enacts it.
        Then, nao's policy would need to be computed in the rollout_trajectories() loop

Then, switch keyboard input to gamepad input
'''




# TODO: extract the winner of the game to add to the message
# TODO: make sure the dialog box shows the current scores
    # maybe also show nao contribution to the scores?
# TODO: test randomly cycling throw the powerups
# TODO: add the enemy double bullet powerup
# TODO: brainstorm behavior tree integrated control flow


'''
Behavior tree control flow:

    - Contraints:
        - The robot should be running the standard behavior tree in the background in case it needs
           to move or monitor surroundings, etc.
        - Rollout interactive trajectory can be adapted into a behavior tree:
            - Behavior node runs the game for a particular segment until pause
            - That node can order speech commands via blackboard?
            - parallel node waiting for speech commands will speak whenever needed

    - Simpler version with no behavior tree
        - Use rollout_interactive_trajectory() as is, but integrate calls to speech api when robot needs to talk or move
        - Same for movements?
        
        - If behavior tree is really needed, just have the entire game as one behavior running in parallel with other
           environment monitoring. Each tick/update() call runs a step of the game while loop, and any required
           speech commands are made inside the behavior's update() function. Pygame clock will maintain desired framerate

           

    - 
           
'''


'''
Behavior tree + pygame interfacing:
* pygame interactive game loop takes in robot action queue as an argument
* behavior tree node that runs the game loop creates this queue, and each add operation updates a blackboard variable
* fairzoo repo becomes a submodule of shutter tarzan, which imports the function to run the interactive game loop




Next behavior tree steps:
* fairzoo: Robot action queue will publish ros messages or write to json file
* tarzan: behavior node update() will check for updates to robot action queue and pass them to blackboard for execution
* queue could be a tiny file for each action so that checking for new actions is trivial/fast?
* or queue could be a single json that will be parsed by behavior node update() function each time


Variable management between env and interactive gameplay loop is getting messy
TODO: merge the dialog box breaks into the environment? or maybe a wrapper class above it?
TODO: game duration from consts is used in featurization in env_utils, in env itself, and in game breaks in rollout function
        need to keep them all in step with each other
TODO: think about defaults at various levels of argument propagation and avoid redundancy
TODO: overly complicated environment variable initiation in create_env?


TODO:TODO:TODO: game duration for state/saving != for ending loop. THIS IS TEMPORARY AND WILL BREAK SAVING AND SCORING


Tutorial changes
- only one player appears on the screen and can act on environment
    * hide non-interactive player and nao in the middle, don't let them act, just no-op their actions
- once sufficient enemy elimination rewards accured then we can terminate
    * state.is_terminal() based on cumulative number of enemies eliminated





'''


'''
Retry logic for each game or each segment of the game?

define constants for remaining event validations e.g. powerups

'''

# Add a pause on cart buttons for instruction from experimenter --> done

# also, extra long wait time on policy setting announcement --> done

# end of game recap should show the score threshold --> done

# remind bonus threshold in temporary waiting screen? --> maybe not

# check on if same person pushes the buttons twice
    # --> same controller controls both ships, can't recover.
    # --> now that's impossible

# alias for the launch command --> done

# screen recording command should append a timestamp to the filename --> done

# visual indicator its not collecting input yet --> done
        


#Changes made agent = 'human' and reward model from full to scores because of parallelization. Fixed boxes on bullets as they were too small, fixed respawnign that happened when non human agents were hit. Added new reward function
class CompetitiveSpaceInvadersEnv(gym.Env):
    metadata = {
        "name": "SpaceInvaders_v0",
        "render_modes": RenderModes.values(), 
        "render_fps": 4
    }

    # TODO: instantiate config outside of env instantiation to pass these all at once
    def __init__(
        self,
        action_space=ActionSpaces.NAO_ONLY,
        verbose=False,
        reward_model=RewardTypes.FULL,
        render_mode=RenderModes.RGB_ARRAY.value,
        support_policy=NaoSupportPolicies.EQUAL_SUPPORT,
        rules_based_human_policy=False,
        minimal_complexity_env=False,
        renderTime=True,
        renderFrameNumber=True,
        use_observations=True,
        interactive_mode=InteractiveModes.NONE,
        interactive_players=[Players.HUMAN, Players.SHUTTER], # Only used if interactive_mode is not NONE
        num_game_segments=1,
        game_duration_frames=DynamicsConsts.MAX_NUM_FRAMES_PER_GAME,
        independent_victory_score_threshold=None,
        game_type=GameTypes.COMPETITIVE,
        # disadvantaged_player=None,
        # disadvantaged_idle_prob=0.0,
        # disadvantaged_extra_shot_cooldown_frames=0,
        human_weak_player_flag = False,
        shutter_weak_player_flag = False
    ):
        """
        Arguments:
            agent -> which agent is being trained
            verbose -> whether or not information is to be returned 
            reward_model -> what reward model the agent is going to train 
        """
        super(CompetitiveSpaceInvadersEnv, self).__init__()

        self.config = SpaceInvadersConfig(
            action_space=action_space,
            verbose=verbose, reward_model=reward_model, render_mode=render_mode, support_policy=support_policy,
            rules_based_human_policy=rules_based_human_policy,
            minimal_complexity_env=minimal_complexity_env,
            use_recent_states_buffer=True,
            use_observations=use_observations,
            interactive_mode=interactive_mode,
            num_game_segments=num_game_segments,
            game_duration_frames=game_duration_frames,
            independent_victory_score_threshold=independent_victory_score_threshold,
            interactive_players=interactive_players,
            game_type=game_type,
            # disadvantaged_player=disadvantaged_player,
            # disadvantaged_idle_prob=disadvantaged_idle_prob,
            # disadvantaged_extra_shot_cooldown_frames=disadvantaged_extra_shot_cooldown_frames,
            shutter_weak_player_flag = shutter_weak_player_flag,
            human_weak_player_flag = human_weak_player_flag
        )
        self.human_weak_player_flag = human_weak_player_flag
        self.shutter_weak_player_flag = shutter_weak_player_flag

        # Set up rendering
        self.render_time = renderTime
        self.render_frame_count = renderFrameNumber
        self.screen, self.rendered_objects = self._init_rendered_objects()

        # Observations are dictionaries 
        assert ObservationConsts.OBS_SPACE_TYPE == 'Box', "Observation space should be of type 'Box'"
        obs_shape = ObservationConsts.SHAPE if not self.config.minimal_complexity_env else ObservationConsts.MINIMAL_COMPLEXITY_SHAPE
        if self.config.use_recent_states_buffer:
            obs_shape = (obs_shape[0] * ObservationConsts.RECENT_STATES_BUFFER_SIZE,)
        self.observation_space = gym.spaces.Box(
            low=ObservationConsts.MIN_FEATURE_VALUE,
            high=ObservationConsts.MAX_FEATURE_VALUE,
            shape=obs_shape,
            dtype=ObservationConsts.DATA_TYPE
        )

        # Human/Shutter agents use the standard gameplay actions. The Nao-only setup
        # instead chooses whether Nao supports Human, supports Shutter, or idles in the center.
        assert ActionConsts.ACTION_SPACE_TYPE == 'Discrete', "Action space should be of type 'Discrete'"
        if self.config.action_space == ActionSpaces.NAO_ONLY:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Discrete(ActionConsts.DIM)

        # Define state variables
        self.state = SpaceInvadersState(self.config, init_all_values=True)
        self.step_info = StepInfo()
        self.last_reward_components = {}
        if self.config.use_recent_states_buffer:
            self.recent_observations: Deque[np.ndarray] = deque(maxlen=ObservationConsts.RECENT_STATES_BUFFER_SIZE)
            self.recent_state_frames: Deque[int] = deque(maxlen=ObservationConsts.RECENT_STATES_BUFFER_SIZE)

        # Agent handling
        if self.config.action_space == ActionSpaces.JOINT_AGENT:
            raise NotImplementedError("Joint agent not yet implemented")
        self.agent_selected = action_space_to_player(self.config.action_space).value  
        self.agents_unselected = [ag for ag in Players.values() if ag != self.agent_selected] # Define unselected agents

        #Nao Policy
        self.nao_policy_instance = NaoPolicy(
            InitialStateConsts.NAO_POSITION_Y, RenderConsts.SCREEN_WIDTH, RenderConsts.SCREEN_HEIGHT, DynamicsConsts.VERTICAL_BUFFER,
            DynamicsConsts.HIT_RANGE, DynamicsConsts.SECOND_HIT_RANGE, DynamicsConsts.SHOOTING_RANGE, DynamicsConsts.NAO_RELATIVE_SPEED, DynamicsConsts.FREQUENCY_BOUND_FRAMES, 
            PolicyConsts.POLICY_MODE, #self.config.support_policy
        )

        # Initialize clock
        self.clock = pygame.time.Clock()
        #self.state.start_time = time.time() # Start time for the game

    # Determine if using virtual display (not shown on screen but allowing pygame to compute placement of visual elements)
    def _using_visible_display(self):
        if self.config.render_mode == RenderModes.RGB_ARRAY.value:
            return False
        elif self.config.render_mode in [RenderModes.DISPLAY_WINDOW.value, RenderModes.DISPLAY_FULLSCREEN.value]:
            return True
        else:
            raise ValueError(f"Unknown render mode: {self.config.render_mode}")
    
    def _init_rendered_objects(self):
        # Init pygame
        pygame.init()

        #if self.config.render_mode == RenderModes.RGB_ARRAY.value:
        if not self._using_visible_display():
            # Set up a virtual screen that will not be displayed
            screen = pygame.Surface(
                (
                    RenderConsts.SCREEN_WIDTH*RenderConsts.RENDER_UPSCALE_FACTOR,
                    RenderConsts.SCREEN_HEIGHT*RenderConsts.RENDER_UPSCALE_FACTOR
                )
            )
            #screen = None
        else:
            # Set up the screen, if a display exists
            try:
                screen = pygame.display.set_mode(
                    (
                        RenderConsts.SCREEN_WIDTH*RenderConsts.RENDER_UPSCALE_FACTOR,
                        RenderConsts.SCREEN_HEIGHT*RenderConsts.RENDER_UPSCALE_FACTOR
                    ),
                    pygame.SCALED
                )
                if self.config.render_mode == RenderModes.DISPLAY_FULLSCREEN.value:
                    pygame.display.toggle_fullscreen()
                print("Successfully setup pygame rendering.")
            except pygame.error as e:
                print(f"Error setting up pygame rendering: {e}")
                print("Skipping rendering.")
                screen = None
        pygame.display.set_caption("Space Invaders")

        # Init rendered objects
        rendered_objects = SpaceInvadersRenderObjects(
            ship_image=load_ship_image(Players.HUMAN.value),
            ship_inactive_image=load_ship_image(Players.HUMAN.value, inactive=True),
            shutter_image=load_ship_image(Players.SHUTTER.value),
            shutter_inactive_image=load_ship_image(Players.SHUTTER.value, inactive=True),
            nao_image=load_ship_image(Players.NAO.value),
            nao_inactive_image=load_ship_image(Players.NAO.value, inactive=True),
            enemy_image=load_ship_image(Enemies.ENEMY1.value),
            enemy_image2=load_ship_image(Enemies.ENEMY2.value),
            bullet_image=load_bullet_image(enemy=False),
            enemy_bullet_image=load_bullet_image(enemy=True),
            font=pygame.font.Font(RenderConsts.FONT_PATH, RenderConsts.FONT_SIZE),
            small_font=pygame.font.Font(RenderConsts.FONT_PATH, RenderConsts.SMALL_FONT_SIZE),
        )
        return screen, rendered_objects
    
    def _update_recent_states_buffer(self, validate=True):
        current_observation = self.state.get_observation(minimal_observation=self.config.minimal_complexity_env)
        self.recent_observations.append(np.array(current_observation, copy=True))
        self.recent_state_frames.append(self.state.time_state.frame)

        if validate:
            for i, frame in enumerate(reversed(self.recent_state_frames)):
                expected_frames_until_current = i
                frames_until_current = self.state.time_state.frame - frame
                if frames_until_current != expected_frames_until_current:
                    raise ValueError(f"Recent state {i} items from the end of the buffer has {frames_until_current} frames until current state. Expected {expected_frames_until_current}")
    
    def _get_stacked_recent_observations(self):
        # gather observation stacked from recent states
        stacked_observation = np.concatenate(list(self.recent_observations), axis=0)

        # pad stacked observation to constant dim
        assert ObservationConsts.OBS_SPACE_TYPE == 'Box', "Observation space should be of type 'Box'"
        expected_obs_dim = self.observation_space.shape[0]
        max_diff = max(0, expected_obs_dim - stacked_observation.size)
        stacked_observation = np.pad(stacked_observation, (max_diff, 0), mode='constant', constant_values=-1)
        
        return stacked_observation
    
    def reset(self, seed=None, options=None):
        """
        Reset game to default values 
        """
        # Reset all variables
        self.state.reset()

        observation = None
        if self.config.use_recent_states_buffer:
            self.recent_observations.clear()
            self.recent_state_frames.clear()
            self._update_recent_states_buffer()
            if self.config.use_observations:
                observation = self._get_stacked_recent_observations()
        else:
            if self.config.use_observations:
                observation = self.state.get_observation(minimal_observation=self.config.minimal_complexity_env)

        info = self._get_info()

        self.clock = pygame.time.Clock() # Reset clock

        return observation, info

    def step(self, action):
        """
        Perform a step in the game 
        """
        #self.agent_selected = 'Human'
        if FORCE_STEP_BY_FRAME_RATE:
            self.clock.tick(DynamicsConsts.FRAMES_PER_SECOND)  # This will ensure that step() doesn't execute more than 60 times per second. without this the game is much faster and 124 fps on my mac. This adjusts it to the original game.
        self.state.time_state.stamp = time.time()  # Update the current time stamp
        
        # Clear step event tracking workspace
        self.step_info = StepInfo()

        # Check if the game is over
        # NOTE: this happens before the environment is updated to indicate that the current state is the final state,
            # and the last actions should be parsed and acted upon to determine the final rewards and state.
            # A properly implemented control loop should not call step() again after this point.
        #Terminate if 7,200 frames (2 minutes) have passed 
        #terminated = self.state.time_state.frame >= self.config.game_duration_frames #10000 Terminate after 7,200 frames (or 2 mins)
        terminated = self.state.is_terminal(self.config)
        
        #Implement actions for all agents. 
        
        # OLD VERSION:
        #left,right,shoot, nearest_enemy_nao= self.nao_policy()
        # if not self.config.minimal_complexity_env:
        #     left, right, shoot, nearest_enemy_nao, nao_supporting_player, agent_powerup = self.nao_policy_instance.nao_policy(
        #         state=self.state, images=self.rendered_objects
        #     )

        #     # if (left == 0) and (right == 0) and (shoot == 0):
        #         # print(nearest_enemy_nao, 'nearest_enemy_nao', self.state.time_state.frame, 'frame', s2s, 's2s')
        #     left_shutter, right_shutter, shoot_shutter = self.shutter_policy(nearest_enemy_nao)

        #     if nao_supporting_player == Players.HUMAN:
        #         self.state.history.support_frame_count[Players.HUMAN] +=1 #add to the frame count if support Human
        #     elif nao_supporting_player == Players.SHUTTER:
        #         self.state.history.support_frame_count[Players.SHUTTER] +=1 #add to the frame count if support Shutter
        #     self.state.players_state.nao_supporting_player = nao_supporting_player
        # else:
        #     left, right, shoot = False, False, False
        #     left_shutter, right_shutter, shoot_shutter = False, False, False


        if self.config.minimal_complexity_env:
            # No actions for Nao and Shutter
            left, right, shoot = False, False, False
            left_shutter, right_shutter, shoot_shutter = False, False, False

        else:
            ##### For old entrypoint before play_interactive_game.py, where action encodes only human/left action
            # # Human action is automated and determined via step(action) in gym style
            # action_human = action

            # # Shutter policy is automated
            # left_shutter, right_shutter, shoot_shutter = self.shutter_policy(nearest_enemy_nao)
            #####
            if action == 0:
                selected_player = Players.HUMAN
            elif action == 1:
                selected_player = Players.SHUTTER
            elif action == 2:
                selected_player = Players.NAO
            else:
                raise ValueError("Invalid Nao selection action")
            
            
            #### For new play_interactive_game.py entrypoint where `action` has ignorable no-ops for non-interactive players
            # TODO: refactor so that automated policy decides the action within play_interactive_game.py+utils instead of passing no-op actions
            left, right, shoot, nearest_enemy_nao, nao_supporting_player, agent_powerup,  = self.nao_policy_instance.nao_policy(
                state=self.state, 
                
                support_target = selected_player, #Figure this out,
                images=self.rendered_objects
            ) #removed requested_robot_action
            if nao_supporting_player == Players.HUMAN:
                self.state.history.support_frame_count[Players.HUMAN] +=1 #add to the frame count if support Human
            elif nao_supporting_player == Players.SHUTTER:
                self.state.history.support_frame_count[Players.SHUTTER] +=1 #add to the frame count if support Shutter
            self.state.players_state.nao_supporting_player = nao_supporting_player
            left_human, right_human, shoot_human = self.human_policy()
            #shoot if the action is shoot and the random value is less than the shooting threshold and there is a human weak player flag
            # if self.human_weak_player_flag and shoot_human:
            #     print('shoot_human before', shoot_human)

            #     shoot_human = random.random() < PlayerPerformanceConsts.SHOOTING_THRESHOLD
            #     print('shoot_human after', shoot_human)

            #otherwise just ignore it
            action_human = boolean_policy_to_index(left_human, right_human, shoot_human)

            left_shutter, right_shutter, shoot_shutter = self.shutter_policy(nearest_enemy_nao)
            # if self.shutter_weak_player_flag and shoot_shutter:
            
            #     shoot_shutter = random.random() < PlayerPerformanceConsts.SHOOTING_THRESHOLD
                    


        # else:
        #     # Run Nao policy and update support decision
        #     left, right, shoot, nearest_enemy_nao, nao_supporting_player, agent_powerup,  = self.nao_policy_instance.nao_policy(
        #         state=self.state, images=self.rendered_objects
        #     ) #removed requested_robot_action
            
        #     if self.config.game_type == GameTypes.COMPETITIVE:
        #         # Only communicate nao's policy if the game is competitive and not tutorial
        #         #self.step_info.requested_robot_action = requested_robot_action
        #         pass
        #     if nao_supporting_player == Players.HUMAN:
        #         self.state.history.support_frame_count[Players.HUMAN] +=1 #add to the frame count if support Human
        #     elif nao_supporting_player == Players.SHUTTER:
        #         self.state.history.support_frame_count[Players.SHUTTER] +=1 #add to the frame count if support Shutter
        #     self.state.players_state.nao_supporting_player = nao_supporting_player

        #     # Run standard shutter and human policy vs control interactively
        #     if self.config.interactive_mode == InteractiveModes.NONE:
        #         ##### For old entrypoint before play_interactive_game.py, where action encodes only human/left action
        #         # # Human action is automated and determined via step(action) in gym style
        #         # action_human = action

        #         # # Shutter policy is automated
        #         # left_shutter, right_shutter, shoot_shutter = self.shutter_policy(nearest_enemy_nao)
        #         #####
                
               
        #         #### For new play_interactive_game.py entrypoint where `action` has ignorable no-ops for non-interactive players
        #         # TODO: refactor so that automated policy decides the action within play_interactive_game.py+utils instead of passing no-op actions
        #         left_human, right_human, shoot_human = self.human_policy()
        #         action_human = boolean_policy_to_index(left_human, right_human, shoot_human)
        #         left_shutter, right_shutter, shoot_shutter = self.shutter_policy(nearest_enemy_nao)
        #     else:
        #         # interactive_players determines which players will be controlled interactively vs using an 
        #         # automated policy. It could be one or both human/shutter that are interactive

        #         if Players.HUMAN in self.config.interactive_players:
        #             # Human player controlled interactively
        #             #left_human, right_human, shoot_human = index_to_boolean_policy(action["left_player_action_index"])
        #             action_human = action["left_player_action_index"]
        #         elif self.config.game_type == GameTypes.COMPETITIVE:
        #             # Human action is automated and determined via step(action) in gym style
        #             left_human, right_human, shoot_human = self.human_policy()
        #             action_human = boolean_policy_to_index(left_human, right_human, shoot_human)
        #         elif self.config.game_type == GameTypes.COMPETITIVE_TUTORIAL:
        #             # Human/left action is a no-op in the tutorial while Shutter/right player is controlled interactively
        #             left_human, right_human, shoot_human = False, False, False
        #             action_human = boolean_policy_to_index(left_human, right_human, shoot_human)

        #         if Players.SHUTTER in self.config.interactive_players:
        #             # Shutter player controlled interactively
        #             left_shutter, right_shutter, shoot_shutter = index_to_boolean_policy(action["right_player_action_index"])
        #         elif self.config.game_type == GameTypes.COMPETITIVE:
        #             # Shutter player controlled by automated policy
        #             left_shutter, right_shutter, shoot_shutter = self.shutter_policy(nearest_enemy_nao)
        #         elif self.config.game_type == GameTypes.COMPETITIVE_TUTORIAL:
        #             # Shutter/right action is a no-op in the tutorial while Human/left player is controlled interactively
        #             left_shutter, right_shutter, shoot_shutter = False, False, False

        # TODO: update to condense left/right/shoot actions into single variable for each player
        self.action_handler(action_human,left,right,shoot,left_shutter, right_shutter, shoot_shutter)

        #Adjust the state at every step
        self.enemy_shot_handler() # Implement enemy shooting actions
        self.update_bullet_positions() # Update the position of bullets 
        self.bullet_threat_check() # Check if any bullets are threatening the player
        self.check_collisions(agent_powerup) # Check if any bullets have collided with enemies or players
        self.human_respawn_handler() # Check for destroyed enemies that need to respawn
        self.enemy_respawn_handler() # Check for destroyed enemies that need to respawn
        self.score_handler() # Updates based on score changes from previous handlers
        rewards = self.reward_handler() # Calculate rewards for the frame
        self.state.time_state.frame += 1 # Update frame count
        #Get Reward
        #print('agent selected',self.agent_selected)
        #reward = self.state.reward_state.rewards[self.agent_selected] #reward
        reward = rewards[Players(self.agent_selected)]

        #Add information   about the state
        info = self._get_info()
        self.state.reward_state.sum_reward += reward
        running_avg_reward = self.state.reward_state.sum_reward/self.state.time_state.frame

        if self.config.verbose:
            print("Time: ", self.state.time_state)

            if self.state.time_state.frame == self.config.game_duration_frames: #Optional: to see how well it is performing as it is training
                print('Episode Rewards:', self.state.reward_state.sum_reward)
                print('average reward:', running_avg_reward)
                print('number of hits:' ,self.state.score_state.human_hit_count)

            if self.state.bullet_state.enemyb_free == []:
                print('WARNING')

        if self.config.use_observations and not self.config.use_recent_states_buffer:
            observation = self.state.get_observation(minimal_observation=self.config.minimal_complexity_env) # Get observation 
        else:
            observation = None
        
        if TEST_OBSERVATION_STATE_CYCLE:
            assert self.config.use_observations, "Observations are not enabled"
            state_recon = SpaceInvadersState.from_observation(observation, self.config, minimal_observation=self.config.minimal_complexity_env)
            observation_recon = state_recon.get_observation(minimal_observation=self.config.minimal_complexity_env)

            assert self.state.equal(state_recon, ignore_missing_fields=True, verbose=True), "State reconstruction failed"
            assert np.all(observation == observation_recon), "Observation reconstruction failed"

        if TEST_JSON_SAVE_CYCLE:
            import json
            from utils import CustomJSONEncoder, CustomJSONDecoder

            state_dict = self.state.to_dict()
            state_str = json.dumps(state_dict, cls=CustomJSONEncoder)
            state_json = json.loads(state_str, object_hook=CustomJSONDecoder.custom_json_decode)
            state_json_recon = SpaceInvadersState(init_all_values=False)
            state_json_recon._from_dict(state_json)
            assert self.state.equal(state_json_recon, ignore_missing_fields=False, verbose=True), "State JSON reconstruction failed"
            state_json_recon2 = SpaceInvadersState.from_dict(state_json)
            assert self.state.equal(state_json_recon2, ignore_missing_fields=False, verbose=True), "State JSON reconstruction failed"

            step_info_dict = self.step_info.to_dict()
            step_info_str = json.dumps(step_info_dict, cls=CustomJSONEncoder)
            step_info_json = json.loads(step_info_str, object_hook=CustomJSONDecoder.custom_json_decode)
            step_info_json_recon = StepInfo.from_dict(step_info_json)
            assert self.step_info == step_info_json_recon, "StepInfo JSON reconstruction failed"
            assert self.step_info.equal(step_info_json_recon, ignore_missing_fields=False, enforce_type_match=False), "StepInfo JSON reconstruction failed"
            assert self.step_info.equal(step_info_json_recon, ignore_missing_fields=False, enforce_type_match=True), "StepInfo JSON reconstruction failed"


        # Clear step event tracking workspace
        self.step_info = StepInfo()

        if self.config.use_recent_states_buffer:
            self._update_recent_states_buffer()
            if self.config.use_observations:
                observation = self._get_stacked_recent_observations()
        
        return observation, reward, terminated, False, info 
    
    def set_boost(self, boost: str):
        # Reset boost for all players
        for player in Players:
            self.state.players_state.double_bullet_boost[player] = False

        # Apply specified boost
        if boost == "left_player_double_bullet":
            self.state.players_state.double_bullet_boost[Players.HUMAN] = True
        elif boost == "right_player_double_bullet":
            self.state.players_state.double_bullet_boost[Players.SHUTTER] = True
        elif boost == "none":
            pass
        else:
            raise ValueError(f"Invalid boost value: {boost}")
        

    
    def human_policy(self):
        if self.config.minimal_complexity_env:
            raise NotImplementedError("Minimal complexity environment doesn't have the argument to decide which minimal human policy")
            # from agents.policies import minimal_human_rules_based_policy_from_state
            # return minimal_human_rules_based_policy_from_state(self.state)
        left, right, shoot = human_rules_based_policy_from_state(self.state)
        return left, right, shoot #self._apply_policy_handicap(Players.HUMAN, left, right, shoot)

    # def _apply_policy_handicap(self, player: Players, left: bool, right: bool, shoot: bool):
    #     if self.config.disadvantaged_player not in [player.value, "both"]:
    #         return left, right, shoot

    #     if (
    #         shoot
    #         and self.config.disadvantaged_extra_shot_cooldown_frames > 0
    #     ):
    #         last_shot_frame = self.state.history.last_shot_frame[player.value]
    #         if last_shot_frame != float("-inf"):
    #             frames_since_last_shot = self.state.time_state.frame - last_shot_frame
    #             if frames_since_last_shot < self.config.disadvantaged_extra_shot_cooldown_frames:
    #                 shoot = False

    #     if self.config.disadvantaged_idle_prob > 0.0 and random.random() < self.config.disadvantaged_idle_prob:
    #         return False, False, False

    #     return left, right, shoot

    # def save_state(self):
    #     """Return a deepcopy of the environment's state."""
    #     return self.state.to_dict()

    #     # # TODO: migrate variables to state object
    #     # state = {
    #     #     "state": self.state.to_dict(),
    #     #     # 'human_position_x': self.human_position_x,
    #     #     # 'human_position_y': self.human_position_y,
    #     #     # 'shutter_position_x': self.shutter_position_x,
    #     #     # 'shutter_position_y': self.shutter_position_y,
    #     #     # 'nao_position_x': self.nao_position_x,
    #     #     # 'nao_position_y': self.nao_position_y,
    #     #     # 'enemies_active': copy.deepcopy(self.enemies_active),
    #     #     # 'scores': copy.deepcopy(self.scores),
    #     #     # 'frame': self.frame,
    #     #     # 'rewards': copy.deepcopy(self.rewards),
    #     #     # 'time': self.time,
    #     #     # 'sum_reward': self.sum_reward,
    #     #     # 'running_avg_reward': self.running_avg_reward,
    #     #     # 'human_hit_count': self.human_hit_count,
    #     #     # Add any other state variables that are necessary
    #     # }
    #     # return state

    # def load_state(self, state):
    #     """Load the environment's state from the given state dictionary."""
    #     self.state = SpaceInvadersState.from_dict(state)

    #     # # TODO: migrate variables to state object
    #     # self.state = SpaceInvadersState.from_dict(state["state"])

    #     # # self.human_position_x = state['human_position_x']
    #     # # self.human_position_y = state['human_position_y']
    #     # # self.shutter_position_x = state['shutter_position_x']
    #     # # self.shutter_position_y = state['shutter_position_y']
    #     # # self.nao_position_x = state['nao_position_x']
    #     # # self.nao_position_y = state['nao_position_y']
    #     # # self.enemies_active = copy.deepcopy(state['enemies_active'])
    #     # # self.scores = copy.deepcopy(state['scores'])
    #     # # self.frame = state['frame']
    #     # # self.rewards = copy.deepcopy(state['rewards'])
    #     # # self.time = state['time']
    #     # # self.sum_reward = state['sum_reward']
    #     # # self.running_avg_reward = state['running_avg_reward']
    #     # # self.human_hit_count = state['human_hit_count']
    #     # # Load any other state variables that are necessary

    def get_state(self, copy_state: bool = True) -> SpaceInvadersState:
        state = self.state
        if copy_state:
            state = copy.deepcopy(state)
        return state

    def set_state(self, state: SpaceInvadersState, copy_state: bool = True):
        if copy_state:
            state = copy.deepcopy(state)
        self.state = state
        
        if self.config.use_recent_states_buffer:
            self.recent_observations.clear()
            self.recent_state_frames.clear()
            self._update_recent_states_buffer()

    def get_recent_states(self, copy_states: bool = True) -> Deque[SpaceInvadersState]:
        raise NotImplementedError("not yet tested")
        states = self.recent_states
        if copy_states:
            states = copy.deepcopy(states)
        return states
    
    def set_recent_states(self, states: Deque[SpaceInvadersState], copy_states: bool = True):
        raise NotImplementedError("not yet tested")
        if copy_states:
            states = copy.deepcopy(states)
        self.recent_states = states
        self.state = states[-1]

    def remove_pygame_objects(self):
        self.screen = None
        self.rendered_objects = None
        self.clock = None

    def restore_pygame_objects(self):
        # self._init_pygame()
        # self._load_images()
        self._init_rendered_objects()
        # TODO: do any other timers need to update to stay in sync with pygame?
        self.clock = pygame.time.Clock()

    def get_player_ship_edges(self, player):
        """Return the left, right, and upper edges of the player's ship."""
        if player == Players.HUMAN:
            x_pos, y_pos = self.state.players_state.human_position_x, self.state.players_state.human_position_y
        elif player == Players.SHUTTER:
            x_pos, y_pos = self.state.players_state.shutter_position_x, self.state.players_state.shutter_position_y
        elif player == Players.NAO:
            x_pos, y_pos = self.state.players_state.nao_position_x, self.state.players_state.nao_position_y
        else:
            raise ValueError(f"Player {player} not recognized")
        
        left_edge = x_pos - DynamicsConsts.SHIP_WIDTH / 2
        right_edge = x_pos + DynamicsConsts.SHIP_WIDTH / 2
        upper_edge = y_pos + DynamicsConsts.SHIP_HEIGHT / 2
        return left_edge, right_edge, upper_edge
    
    def shutter_policy(self,nearest_enemy_nao):
        """
        Policy for Shutter. Identifies nearest bullets and enemies and adjusts its position or shoots.
        """
        #initialize variables for actions
        left = False
        right = False
        shoot = False
        hit = False
        nearest_bullet = [0,0]

        x_diff_prev= 1 #RenderConsts.SCREEN_WIDTH/RenderConsts.SCREEN_WIDTH 

        #Find nearest bullet only in the right side of the screen
        bullets_left_side, bullets_right_side = self.state.get_enemy_bullets_by_side()
        for bullet in bullets_right_side:
            x_diff = abs(bullet[0]-self.state.players_state.shutter_position_x)
            #if bullet[1]< InitialStateConsts.SHUTTER_POSITION_Y and bullet[1]> DynamicsConsts.VERTICAL_BUFFER and x_diff<DynamicsConsts.HIT_RANGE *2:
            if bullet[1]> DynamicsConsts.VERTICAL_BUFFER and x_diff<DynamicsConsts.HIT_RANGE *2:
                if x_diff < x_diff_prev:
                    nearest_bullet= bullet
                    x_diff_prev = x_diff


        #look through active enemies and assign them based on whether they are on the left or right side
        # self.state.enemy_state.active_enemies_right_side.clear()
        # self.state.enemy_state.active_enemies_left_side.clear()
        # for (x, y), active in zip(zip(InitialStateConsts.ENEMIES_X, InitialStateConsts.ENEMIES_Y), self.state.enemy_state.enemies_active):
        #     if active:
        #         if x >= 0.5:
        #             self.state.enemy_state.active_enemies_right_side.append((x,y))
        #         else:
        #             self.state.enemy_state.active_enemies_left_side.append((x,y))
        # active_enemies_left_side, active_enemies_right_side = self.state.get_active_enemies_by_side()

        # Find nearest enemy by Y
        # nearest_enemy = [0,0]
        # nearest_x_diff = 1 #RenderConsts.SCREEN_WIDTH
        # nearest_y_diff = DynamicsConsts.VERTICAL_BUFFER
        # closest_x_diff =1 # RenderConsts.SCREEN_WIDTH

        # #In the original version they looked through each enemy and checked if it was active or not. We are just going to check active enmies. 
        # #Find nearest enemy for Shutter Ship
        # for enemy in active_enemies_right_side:
        #     if enemy == nearest_enemy_nao:
        #         pass
        #     else:
        #         check_distance_x = abs(enemy[0] - self.state.players_state.shutter_position_x)
        #         check_distance_y = abs(enemy[1] - InitialStateConsts.SHUTTER_POSITION_Y)

        #         #index = self.state.enemy_state.active_enemies_right_side.index(enemy)
        #         if check_distance_y < nearest_y_diff:
        #             nearest_enemy = enemy
        #             nearest_x_diff = check_distance_x
        #             nearest_y_diff = check_distance_y
        #         elif check_distance_y == nearest_y_diff and check_distance_x < nearest_x_diff:
        #             nearest_enemy = enemy
        #             nearest_x_diff = check_distance_x
        #             nearest_y_diff = check_distance_y
        #         #elif check_distance_y < (nearest_y_diff+(50/RenderConsts.SCREEN_HEIGHT)) and check_distance_x < closest_x_diff :
        #         elif check_distance_y < (nearest_y_diff+(DynamicsConsts.SHIP_HEIGHT)) and check_distance_x < closest_x_diff :
        #             closest_x_diff = check_distance_x

        nearest_enemy, (nearest_x_diff, nearest_y_diff, closest_x_diff) = (
            self.state.get_nearest_enemy(
                player=Players.SHUTTER,
                side_to_search="right",
                excluded_enemies=[],#excluded_enemies=[nearest_enemy_nao],
                return_diffs=True,
            )
        )
        
        # check if agent is in danger of being hit by bullet
        if nearest_bullet[0] <= self.state.players_state.shutter_position_x + DynamicsConsts.HIT_RANGE and nearest_bullet[0] >= self.state.players_state.shutter_position_x - DynamicsConsts.HIT_RANGE:
            hit = True

        approached_enemy = False

        if not self.state.can_shoot(Players.SHUTTER):
            #If agent cannot shoot, adjust position
            if hit:
                # TODO: what's 75?
                if nearest_bullet[0] >= 1 - (75/RenderConsts.SCREEN_WIDTH):
                    left = True
                # TODO: what's 55?
                elif nearest_bullet[0] <= DynamicsConsts.MIN_X + (55/RenderConsts.SCREEN_WIDTH):
                    right = True
                elif nearest_bullet[0] > self.state.players_state.shutter_position_x:
                    left = True
                elif nearest_bullet[0] <= self.state.players_state.shutter_position_x:
                    right = True
        else:
            #IF agent CAN SHOOT
            #print('shutter near',nearest_x_diff, 'shootRange', DynamicsConsts.SHOOTING_RANGE, 'closest',closest_x_diff )
            # if nearest_x_diff <= DynamicsConsts.SHOOTING_RANGE or closest_x_diff <= DynamicsConsts.SHOOTING_RANGE:
            #     shoot = True
            if abs(nearest_enemy[0] - self.state.players_state.shutter_position_x) <= DynamicsConsts.SHOOTING_RANGE:
                shoot = True

            if nearest_enemy[0] < self.state.players_state.shutter_position_x:
                if not(nearest_bullet[0] < self.state.players_state.shutter_position_x and InitialStateConsts.SHUTTER_POSITION_Y - nearest_bullet[1] < (200 / RenderConsts.SCREEN_HEIGHT) and self.state.players_state.shutter_position_x - nearest_bullet[0] <= DynamicsConsts.SECOND_HIT_RANGE):
                    left = True
                    approached_enemy = True
            elif nearest_enemy[0] > self.state.players_state.shutter_position_x:
                if not(nearest_bullet[0] > self.state.players_state.shutter_position_x and InitialStateConsts.SHUTTER_POSITION_Y - nearest_bullet[1] < (200 / RenderConsts.SCREEN_HEIGHT) and nearest_bullet[0] - self.state.players_state.shutter_position_x <= DynamicsConsts.SECOND_HIT_RANGE):
                    right = True
                    approached_enemy = True
                    
            if not approached_enemy and hit:
                if x_diff_prev > (5/RenderConsts.SCREEN_WIDTH):
                    # if not going to hit bullet, don't shoot so you can move
                    shoot = False
                # figure out which way to move so you don't get hit  
                if nearest_bullet[0] >= 1-(75/RenderConsts.SCREEN_WIDTH):
                    left = True
                elif nearest_bullet[0] <= DynamicsConsts.MIN_X + (55/RenderConsts.SCREEN_WIDTH):
                    right = True
                elif nearest_bullet[0] > self.state.players_state.shutter_position_x:
                    left = True
                elif nearest_bullet[0] <= self.state.players_state.shutter_position_x:
                    right = True

        # TODO: can the agent execute multiple actions at once???
        #assert sum([left, right, shoot]) <= 1, "Agent can only execute one action at a time"
        return left, right, shoot #self._apply_policy_handicap(Players.SHUTTER, left, right, shoot)


    def bullet_threat_check(self):
        """"
        Checks whether the Human or Shutter spaceship is aligned in the same column as an enemy bullet and how long until collision.
        """
        def update_player_threats(player, column_offset, enemy_x_positions, threat_attr, collision_attr):
            setattr(self.state.bullet_state, threat_attr, [0] * StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE)
            setattr(self.state.bullet_state, collision_attr, [None] * StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE)

            spaceship_left_edge, spaceship_right_edge, spaceship_upper_edge = self.get_player_ship_edges(player)
            spaceship_left_edge = max(0, spaceship_left_edge)

            threat_values = getattr(self.state.bullet_state, threat_attr)
            collision_values = getattr(self.state.bullet_state, collision_attr)

            for col in range(StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE):
                min_frames_until_collision = None
                bullet_x = enemy_x_positions[col]
                bullet_col = col + column_offset
                for bullet_index in range(StateConsts.MAX_NUM_ENEMY_BULLETS_PER_COLUMN):
                    bullet_y = self.state.bullet_state.bullet_y_positions[bullet_col, bullet_index]
                    if bullet_y > 0 and spaceship_left_edge <= bullet_x <= spaceship_right_edge:
                        distance_to_travel = spaceship_upper_edge - bullet_y
                        if distance_to_travel > 0:
                            threat_values[col] = 1
                            frames_to_collision = distance_to_travel / DynamicsConsts.BULLET_VELOCITY
                            if min_frames_until_collision is None or frames_to_collision < min_frames_until_collision:
                                min_frames_until_collision = frames_to_collision

                if min_frames_until_collision is None:
                    collision_values[col] = ObservationConsts.NO_FRAMES_UNTIL_BULLET_COLLISION
                else:
                    collision_values[col] = min_frames_until_collision

        update_player_threats(
            player=Players.HUMAN,
            column_offset=0,
            enemy_x_positions=InitialStateConsts.HUMAN_ENEMIES_X,
            threat_attr="bullet_threat_binary",
            collision_attr="frames_until_collision",
        )
        update_player_threats(
            player=Players.SHUTTER,
            column_offset=StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE,
            enemy_x_positions=InitialStateConsts.SHUTTER_ENEMIES_X,
            threat_attr="shutter_bullet_threat_binary",
            collision_attr="shutter_frames_until_collision",
        )
    
    def _get_info(self):
        """
        Return structured dictionary state information for troubleshooting
        """
        # info = {
        #     'player_positions': {
        #         'Human': [self.state.players_state.human_position_x, self.state.players_state.human_position_y],
        #         'Shutter': [self.state.players_state.shutter_position_x, self.state.players_state.shutter_position_y],
        #         'Nao': [self.state.players_state.nao_position_x, self.state.players_state.nao_position_y],
        #     },
        #     'enemy_active': self.state.enemy_state.enemies_active,
        #     'bullet_positions': self.state.bullet_state.player_bullets, #check
        #     'enemy_bullet_positions': self.state.bullet_state.enemy_bullets, #check
        #     'scores': self.state.score_state.scores,
        #     'frame': self.state.time_state.frame,
        #     }

        # assuming terminated means success
        success = self.state.time_state.frame >= self.config.game_duration_frames 
        info = {
            "step_info": self.step_info,
            "is_success": success,
            "reward_components": self.last_reward_components,
        }
        
        # TODO: step info should have actions of agents that are part of the environment
        
        return info
    
    # TODO: update this for keyboard controls with agent selected/unselected
    def action_handler(self,action,left_nao,right_nao,shoot_nao,left_shutter, right_shutter, shoot_shutter):
        """
        Handle all agent actions 
        """
        enacted_left_human, enacted_right_human, enacted_shoot_human = False, False, False
        enacted_left_shutter, enacted_right_shutter, enacted_shoot_shutter = False, False, False
        enacted_left_nao, enacted_right_nao, enacted_shoot_nao = False, False, False

        # Human Agent
        left_human,right_human,shoot_human = index_to_boolean_policy(action)
        left_human,right_human,shoot_human = tie_breaker_actions(left_human,right_human,shoot_human)
        if left_human:
            enacted_left_human = self.move_left("Human")
            #print('move left human', enacted_left_human)
            
        if right_human:  #  '1' corresponds to 'move_right'
            enacted_right_human = self.move_right("Human")
            #print('move right human', enacted_right_human)

            
        if shoot_human:  # '2' corresponds to 'shoot'
            enacted_shoot_human = self.shoot("Human")
            #print(' shoot human', enacted_shoot_human)

            
            
        # For Shutter
        for agent in self.agents_unselected:
            if agent == "Shutter":

                #Check that only one action is taken
                left_shutter,right_shutter,shoot_shutter= tie_breaker_actions(left_shutter,right_shutter,shoot_shutter)
                # if left_shutter and shoot_shutter or right_shutter and shoot_shutter or left_shutter and right_shutter:

                #     "error"
                #     assert False,"mix"
                if left_shutter:  
                    enacted_left_shutter = self.move_left(agent)

                if right_shutter:
                    enacted_right_shutter = self.move_right(agent)

                if shoot_shutter:
                    enacted_shoot_shutter = self.shoot(agent)

        # Nao actions
        left_nao,right_nao,shoot_nao= tie_breaker_actions(left_nao,right_nao,shoot_nao)
        if left_nao:
            enacted_left_nao = self.move_left("Nao")
        
        if right_nao:
            enacted_right_nao = self.move_right("Nao")

        if shoot_nao:
            enacted_shoot_nao = self.shoot("Nao")
        
        self.step_info.actions_taken[Players.HUMAN] = SingleAgentAction(enacted_left_human, enacted_right_human, enacted_shoot_human)
        self.step_info.actions_taken[Players.SHUTTER] = SingleAgentAction(enacted_left_shutter, enacted_right_shutter, enacted_shoot_shutter)
        self.step_info.actions_taken[Players.NAO] = SingleAgentAction(enacted_left_nao, enacted_right_nao, enacted_shoot_nao)

    def move_left(self, agent):
        """
        Complete move left action
        """
        if agent not in Players.values():
            raise ValueError(f"Agent {agent} not recognized")
        
        if agent == 'Human' and self.state.players_state.active['Human']:
            #new_position = np.clip(self.state.players_state.human_position_x - 0.00625, 0.0002, 1)
            new_position = np.clip(self.state.players_state.human_position_x - 0.00625, DynamicsConsts.HUMAN_LEFT_LIMIT, DynamicsConsts.HUMAN_RIGHT_LIMIT)
            if new_position == self.state.players_state.human_position_x:
                return False
            self.state.players_state.human_position_x = new_position
            return True
        elif agent == 'Shutter' and self.state.players_state.active['Shutter']:
            #new_position = np.clip(self.state.players_state.shutter_position_x - 0.00625, .5, 1)
            new_position = np.clip(self.state.players_state.shutter_position_x - 0.00625, DynamicsConsts.SHUTTER_LEFT_LIMIT, DynamicsConsts.SHUTTER_RIGHT_LIMIT)
            if new_position == self.state.players_state.shutter_position_x:
                return False
            self.state.players_state.shutter_position_x = new_position
            return True
        elif agent == 'Nao' and self.state.players_state.active['Nao']:
            #new_position = np.clip(self.state.players_state.nao_position_x - 0.00625, 0, 1)
            new_position = np.clip(self.state.players_state.nao_position_x - 0.00625, DynamicsConsts.NAO_LEFT_LIMIT, DynamicsConsts.NAO_RIGHT_LIMIT)
            if new_position == self.state.players_state.nao_position_x:
                return False
            self.state.players_state.nao_position_x = new_position
            return True
        return False

    def move_right(self, agent):
        """
        Complete move right action
        """
        if agent not in Players.values():
            raise ValueError(f"Agent {agent} not recognized")
        
        if agent == 'Human' and self.state.players_state.active['Human']:
            #new_position = np.clip(self.state.players_state.human_position_x + 0.00625, 0, .35)
            new_position = np.clip(self.state.players_state.human_position_x + 0.00625, DynamicsConsts.HUMAN_LEFT_LIMIT, DynamicsConsts.HUMAN_RIGHT_LIMIT)
            if new_position == self.state.players_state.human_position_x:
                return False
            self.state.players_state.human_position_x = new_position
            return True
        elif agent == 'Shutter' and self.state.players_state.active['Shutter']:
            #new_position = np.clip(self.state.players_state.shutter_position_x + 0.00625, 0, 1)
            new_position = np.clip(self.state.players_state.shutter_position_x + 0.00625, DynamicsConsts.SHUTTER_LEFT_LIMIT, DynamicsConsts.SHUTTER_RIGHT_LIMIT)
            if new_position == self.state.players_state.shutter_position_x:
                return False
            self.state.players_state.shutter_position_x = new_position
            return True
        elif agent == 'Nao' and self.state.players_state.active['Nao']:
            #new_position = np.clip(self.state.players_state.nao_position_x + 0.00625, 0, 1)
            new_position = np.clip(self.state.players_state.nao_position_x + 0.00625, DynamicsConsts.NAO_LEFT_LIMIT, DynamicsConsts.NAO_RIGHT_LIMIT)
            if new_position == self.state.players_state.nao_position_x:
                return False
            self.state.players_state.nao_position_x = new_position
            return True
        return False

    def shoot(self, agent): 
        """
        Complete shoot aciton
        """
        # current_time = time.time() * 1000  # Current time in milliseconds
        # time_since_last_shot = current_time - self.state.history.last_shot_time[agent]
        frames_since_last_shot = self.state.time_state.frame - self.state.history.last_shot_frame[agent]

        # Throttling shots based on agent type
        if agent == 'Human' and not self.state.can_shoot(Players.HUMAN):
            return False
        
        if agent == 'Shutter' and not self.state.can_shoot(Players.SHUTTER):
            return False
        
        if agent == 'Nao' and not self.state.can_shoot(Players.NAO):
            return False

        # bullets move upwards at a fixed rate
        bullet_speed = -0.01
        #Create slots for bullets
        # TODO: make this even for both players???
        agent_slots = {
        'Human': [0, 1, 2],
        'Shutter': [3, 4, 5],
        'Nao': [6, 7, 8, 9]
        }
        # Check that the assigned slots line up with the enemy columns
        assert set(range(StateConsts.NUM_ENEMY_COLUMNS)) == {index for sublist in agent_slots.values() for index in sublist}
        # NOTE: this assumes a max of 10 player bullets, across all players. Human gets 4, shutter 3, nao 3

        # Find the lowest available index within the agent's designated slots
        available_slots = [idx for idx in agent_slots[agent] if idx in self.state.bullet_state.playerb_free]

        fired_bullet = False
        if self.state.bullet_state.playerb_free:
            if not available_slots:
                #print("No available slots for", agent)
                pass

            if available_slots:
                #print('able to shoot')
                #if under the threshold then don't shoot. Adding this here because jsut because it can shoot doesn't necessarily mean it will because there may not be sufficient bullet slots 
                if agent == 'Shutter' and self.shutter_weak_player_flag and random.random() >= PlayerPerformanceConsts.SHOOTING_THRESHOLD:
                    return False
                if  agent == 'Human' and self.human_weak_player_flag and random.random() >= PlayerPerformanceConsts.SHOOTING_THRESHOLD:
                    #print('not shooting')
                    return False
                index =  min(available_slots)  # Get the lowest available slot
                self.state.bullet_state.player_bullets[index] = [self.get_agent_position(agent)[0], self.get_agent_position(agent)[1] + bullet_speed, agent]
                self.state.bullet_state.playerb_used.append(index)
                if self.state.players_state.double_bullet_boost[Players(agent)]:
                    self.state.bullet_state.player_multi_bullet_counts[index] = 2
                else:
                    self.state.bullet_state.player_multi_bullet_counts[index] = 1
                    
                # self.state.history.last_shot_time[agent] = current_time  # Correctly updating here
                self.state.history.last_shot_frame[agent] = self.state.time_state.frame
                self.state.bullet_state.playerb_free.remove(index)
                fired_bullet = True

                self.step_info.player_shoot_rewards[Players(agent)] += RewardConsts.PLAYER_SHOOT_REWARD
            #Only track the last five bullets
            self.update_shot_time_history(agent, frames_since_last_shot)
        else:
            if self.config.verbose:
                print("No Free Bullet Slots")
        #print('fired a bullet',fired_bullet)
        return fired_bullet


    def update_shot_time_history(self, agent, frames_since_last_shot):
        """
        Ensure the bullet history doesn't exceed 5 entries. Only tracking the last five bullets
        """ 
        # if len(self.state.history.shot_time_history[agent]) >= 5:
        #     self.state.history.shot_time_history[agent].pop(0)
        # self.state.history.shot_time_history[agent].append(time_since_last_shot)
        # self.state.history.average_shot_frequency[agent] = sum(self.state.history.shot_time_history[agent]) / len(self.state.history.shot_time_history[agent])

        if len(self.state.history.shot_frame_history[agent]) >= 5:
            self.state.history.shot_frame_history[agent].pop(0)
        self.state.history.shot_frame_history[agent].append(frames_since_last_shot)
        self.state.history.average_shot_frequency_frames[agent] = sum(self.state.history.shot_frame_history[agent]) / len(self.state.history.shot_frame_history[agent])

    def get_agent_position(self, agent):
        """
        Given the name of an agent, return the position of said agent.
        """ 
        if agent == 'Human':
            return (self.state.players_state.human_position_x, self.state.players_state.human_position_y)
        if agent == 'Nao':
            return (self.state.players_state.nao_position_x, self.state.players_state.nao_position_y)
        if agent == 'Shutter':
            return (self.state.players_state.shutter_position_x, self.state.players_state.shutter_position_y)

    # TODO: score handler should handle all score updates, and check collisions should just log step info events
    # TODO: this is out of date because it assumes a competitive single victor
    def score_handler(self):
        # + 1 because frame hasn't been incremented yet
        #is_terminal_step = self.state.time_state.frame + 1 >= self.config.game_duration_frames

        if self.state.is_terminal(config=self.config):
            leading_player = self.state.get_leading_score_player(threshold=0)
            if leading_player == Players.HUMAN:
                self.step_info.player_victory_rewards[Players.HUMAN] += RewardConsts.VICTORY_REWARD
            elif leading_player == Players.SHUTTER:
                self.step_info.player_victory_rewards[Players.SHUTTER] += RewardConsts.VICTORY_REWARD
            else:
                raise ValueError(f"Leading player {leading_player} not recognized. Must be {Players.HUMAN} or {Players.SHUTTER}")

    def _get_fairness_threshold(self) -> float:
        if self.config.independent_victory_score_threshold is not None:
            return float(self.config.independent_victory_score_threshold)
        return float(ScoreConsts.BONUS_FOR_HITTING_ENEMY * StateConsts.NUM_ENEMIES / StateConsts.NUM_SIDES)

    def _compute_nao_fairness_reward(self) -> float:
        threshold = self._get_fairness_threshold()
        epsilon = FairnessRewardConsts.EPSILON

        human_progress = self.state.get_player_score(Players.HUMAN, include_robot_contributions=True) / threshold
        shutter_progress = self.state.get_player_score(Players.SHUTTER, include_robot_contributions=True) / threshold

        human_score = self.state.get_player_score(Players.HUMAN, include_robot_contributions=True)
        shutter_score = self.state.get_player_score(Players.SHUTTER, include_robot_contributions=True)
        score_gap = abs(human_score - shutter_score)
        team_reward = 1.0 - min(score_gap / RewardConsts.LEADING_SCORE_THRESHOLD, 1.0)

        progress_denom = human_progress + shutter_progress + epsilon
        progress_equity = (human_progress - shutter_progress) / progress_denom
        outcome_reward = -abs(progress_equity)

        human_support_frames = self.state.history.support_frame_count[Players.HUMAN]
        shutter_support_frames = self.state.history.support_frame_count[Players.SHUTTER]
        total_support_frames = human_support_frames + shutter_support_frames
        if total_support_frames == 0:
            time_equity = 0.0
        else:
            time_equity = (human_support_frames - shutter_support_frames) / (total_support_frames + epsilon)
        time_reward = -abs(time_equity)

        human_reached_threshold = self.state.get_player_score(Players.HUMAN, include_robot_contributions=True) >= threshold
        shutter_reached_threshold = self.state.get_player_score(Players.SHUTTER, include_robot_contributions=True) >= threshold
        if human_reached_threshold and shutter_reached_threshold:
            threshold_reward = 1.0
        elif human_reached_threshold or shutter_reached_threshold:
            threshold_reward = -2.0
        else:
            threshold_reward = 0.0

        total_reward = (
            FairnessRewardConsts.TEAM_WEIGHT * team_reward
            + FairnessRewardConsts.OUTCOME_WEIGHT * outcome_reward
            # + FairnessRewardConsts.TIME_WEIGHT * time_reward
            + FairnessRewardConsts.THRESHOLD_WEIGHT * threshold_reward
        )
        self.last_reward_components = {
            "team": team_reward,
            "outcome": outcome_reward,
            "time": time_reward,
            "threshold": threshold_reward,
            "total": total_reward,
        }
        return total_reward
    
    def reward_handler(self):
        """
        Calculate rewards for each agent.
        Add new reward models for reward tuning experiments.
        """
        if self.config.reward_model == RewardTypes.FULL:
            # Calculate rewards for each agent
            player_rewards = {}
            for player in Players:
                enemy_elimination_reward = self.step_info.enemy_elimination_rewards.get(player, 0)
                player_hit_reward = self.step_info.player_hit_rewards.get(player, 0)
                player_taking_score_lead_reward = self.step_info.player_taking_score_lead_rewards.get(player, 0)
                player_victory_reward = self.step_info.player_victory_rewards.get(player, 0)
                player_rewards[player] = enemy_elimination_reward + player_hit_reward + player_taking_score_lead_reward + player_victory_reward
            return player_rewards
        
        elif self.config.reward_model == RewardTypes.MINIMAL_COMPLEXITY:
            # Calculate rewards for each agent
            player_rewards = {}
            for player in Players:
                enemy_elimination_reward = self.step_info.enemy_elimination_rewards.get(player, 0)
                player_shoot_reward = self.step_info.player_shoot_rewards.get(player, 0)
                player_rewards[player] = enemy_elimination_reward + player_shoot_reward
            return player_rewards

        elif self.config.reward_model == RewardTypes.NAO_FAIRNESS:
            player_rewards = Players.default_dict(float)
            player_rewards[Players.NAO] = self._compute_nao_fairness_reward()
            return player_rewards
        
        else:
            raise ValueError(f"Reward model {self.config.reward_model} not recognized")

        # NOTE: this only adds reward for the human, even though all rewards are indexed separately. Doesn't seem like we need this.
        # else: 
        #     #factoring score increase
        #     for agent in Players.values():
        #         if self.state.players_state.active[agent]:
        #             # Add rewards for hitting an enemy
        #             if agent == 'Human':
        #                 self.state.reward_state.rewards[agent] += (1*self.state.score_state.single_turn_score['Human']) + (-1 * self.state.score_state.single_turn_score['Shutter']) +self.state.reward_state.elimination_reward['Human']
        #         self.state.reward_state.elimination_reward[agent] = 0


    def enemy_shot_handler(self):
        """
        Handle enemy shots
        """
        self.state.bullet_state.shoot_counter += 1
        if not self.config.minimal_complexity_env:
            # Why is this defined for all enemies instead of for each enemy?
            if self.state.bullet_state.shoot_counter >= DynamicsConsts.SHOOT_FREQUENCY: #Enemy can't shoot unles the shoot counter is greater than shoot frequency
                self.enemy_shoot()
                self.state.bullet_state.shoot_counter = 0  # Reset the counter after shooting

    def enemy_shoot(self):
        # Shoot from both sides, but in random order to prevent order effects skewing available bullets
        # TODO: if bullet limits are defined per side instead of globally then this would be no longer necessary?
        if random.choice([True, False]):
            self.enemy_shoot_side("left")
            self.enemy_shoot_side("right")
        else:
            self.enemy_shoot_side("right")
            self.enemy_shoot_side("left")
    
    def enemy_shoot_side(self, side_to_shoot="all"):
        bullet_speed = DynamicsConsts.BULLET_VELOCITY #0.00277
        #current_time = time.time()
        # TODO: what's 0.7 and 1.0?
        #random_delay = random.uniform(0.7, 1.0) #A delay is randomly selected
        if PolicyConsts.POLICY_MODE == "shooting":
            #random_delay_frames = random.uniform(seconds_to_frames(0.7), seconds_to_frames(1.0))
            random_delay_frames=0
        else:
            random_delay_frames =0
            #random_delay_frames = random.uniform(seconds_to_frames(0.1), seconds_to_frames(.1))


        # if (current_time - self.state.history.last_enemy_shot_time) < random_delay:
        #     return
        if (self.state.time_state.frame - self.state.history.last_enemy_shot_frame) < random_delay_frames:

            return

        if not self.state.bullet_state.enemyb_free:
            if self.config.verbose:
                print("No bullets available to shoot.")
            return

        # Get the active enemies 
        active_columns = {}
        for i, (x, y, active) in enumerate(zip(InitialStateConsts.ENEMIES_X, InitialStateConsts.ENEMIES_Y, self.state.enemy_state.enemies_active)):
            if side_to_shoot == "left":
                on_side = x in InitialStateConsts.HUMAN_ENEMIES_X
            elif side_to_shoot == "right":
                on_side = x in InitialStateConsts.SHUTTER_ENEMIES_X
            elif side_to_shoot == "all":
                on_side = True
            else:
                raise ValueError(f"side_to_shoot {side_to_shoot} not recognized. Must be 'left', 'right', or 'all'.")
            
            if not on_side:
                continue

            if active:
                col_idx = InitialStateConsts.ENEMY_IDX_TO_COL_IDX[i]
                if col_idx not in active_columns or y > active_columns[col_idx][1]:
                    active_columns[col_idx] = (i, y)        
        
        if active_columns:
            chosen_col_idx = random.choice(list(active_columns.keys()))

            active_bullets_left, active_bullets_right = self.state.get_enemy_bullets_by_side()
            chosen_col_side = "left" if chosen_col_idx < StateConsts.NUM_ENEMY_COLUMNS_PER_SIDE else "right"
            if chosen_col_side == "left":
                active_bullets_on_side = active_bullets_left
            elif chosen_col_side == "right":
                active_bullets_on_side = active_bullets_right

            # Check if there are too many bullets in the column
            if len(active_bullets_on_side) >= DynamicsConsts.MAX_NUM_BULLETS_PER_SIDE:
                if self.config.verbose:
                    print(f"Too many bullets in column {chosen_col_idx}. Skipping shoot.")
                return

            lowest_enemy_idx, lowest_enemy_y = active_columns[chosen_col_idx]
            if self.state.bullet_state.enemyb_free:
                # if at least one slot is available
                if np.any(self.state.bullet_state.bullet_y_positions[chosen_col_idx, :] == 0):
                
                    bullet_idx = self.state.bullet_state.enemyb_free.pop(0)
                    enemy_x = InitialStateConsts.ENEMIES_X[lowest_enemy_idx]
                    enemy_y = InitialStateConsts.ENEMIES_Y[lowest_enemy_idx] + bullet_speed
                    self.state.history.last_enemy_shot_frame = self.state.time_state.frame
                    self.state.bullet_state.enemy_bullets[bullet_idx] = [enemy_x, enemy_y]
                    self.state.bullet_state.enemyb_used.append(bullet_idx)

                    for slot in range(StateConsts.MAX_NUM_ENEMY_BULLETS_PER_COLUMN):
                        if self.state.bullet_state.bullet_y_positions[chosen_col_idx, slot] == 0:
                            self.state.bullet_state.bullet_y_positions[chosen_col_idx, slot] = enemy_y
                            break
                else:
                    if self.config.verbose:
                        print("No free slots available in column.")
            else:
                if self.config.verbose:
                    print("No free global bullets available.")
        else:
            if self.config.verbose:
                print("No active columns with enemies to shoot from.")


    def update_bullet_positions(self):
        """
        Update positions of all bullets
        """
        bullet_speed = DynamicsConsts.HUMAN_BULLET_VELOCITY #-0.00972  # Human bullet speed
        enemy_bullet_speed = DynamicsConsts.BULLET_VELOCITY  #0.00277  # Enemy bullet speed

        # Update player bullets
        for i in self.state.bullet_state.playerb_used[:]:  # Copy list to avoid modification issues during iteration
            self.state.bullet_state.player_bullets[i][1] += bullet_speed
            # Remove out-of-range player bullets
            if self.state.bullet_state.player_bullets[i][1] > 1 or self.state.bullet_state.player_bullets[i][1] < 0:
                self.state.bullet_state.player_bullets[i] = [0, 0, None]
                self.state.bullet_state.playerb_used.remove(i)
                self.state.bullet_state.playerb_free.append(i)

        # Update enemy bullets
        for i in self.state.bullet_state.enemyb_used[:]:  # Copy list to avoid modification issues during iteration
            self.state.bullet_state.enemy_bullets[i][1] += enemy_bullet_speed 
            # Remove out-of-range enemy bullets
            if self.state.bullet_state.enemy_bullets[i][1] > 1.0 or self.state.bullet_state.enemy_bullets[i][1] < 0:
                self.state.bullet_state.enemy_bullets[i] = [0., 0.]
                self.state.bullet_state.enemyb_used.remove(i)
                self.state.bullet_state.enemyb_free.append(i)

        # Update bullet y positions for all slots, separate from enemy bullet updates
        for col in range(StateConsts.NUM_ENEMY_COLUMNS):
            for slot in range(StateConsts.MAX_NUM_ENEMY_BULLETS_PER_COLUMN):  # Assuming 2 slots per column
                if self.state.bullet_state.bullet_y_positions[col, slot] != 0:
                    self.state.bullet_state.bullet_y_positions[col, slot] += enemy_bullet_speed
                    if self.state.bullet_state.bullet_y_positions[col, slot] > 1.0 or self.state.bullet_state.bullet_y_positions[col, slot] < 0:
                        self.state.bullet_state.bullet_y_positions[col, slot] = 0  # Reset if out of bounds

        # Sort bullets by distance - only if needed
        self.sort_enemy_bullets_by_distance()

    def sort_enemy_bullets_by_distance(self):
        """
        Sorts enemy bullets by their Euclidean distance to the Human and Shutter spaceships.
        """
        def update_player_bullet_distances(player, bullet_side, distance_attr, small_distance_attr, leftmost_enemy_x, rightmost_enemy_x):
            if player == Players.HUMAN:
                player_pos = np.array([self.state.players_state.human_position_x, self.state.players_state.human_position_y])
            elif player == Players.SHUTTER:
                player_pos = np.array([self.state.players_state.shutter_position_x, self.state.players_state.shutter_position_y])
            else:
                raise ValueError(f"Unsupported player for bullet-distance features: {player}")
            bullets_on_side = np.copy(self.state.bullet_state.enemy_bullets)
            if bullet_side == "left":
                bullets_on_side[bullets_on_side[:, 0] > 0.5] = 0
            else:
                bullets_on_side[bullets_on_side[:, 0] < 0.5] = 0

            differences = bullets_on_side - player_pos
            x_differences = differences[:, 0]
            distances = np.linalg.norm(bullets_on_side - player_pos, axis=1)

            directional_distances = distances * np.sign(x_differences)
            directional_distances[distances >= DynamicsConsts.MAX_BULLET_DISTANCE_TO_INCLUDE] = 0
            directional_distances[(bullets_on_side[:, 0] == leftmost_enemy_x)] = -abs(directional_distances[(bullets_on_side[:, 0] == leftmost_enemy_x)])
            directional_distances[(bullets_on_side[:, 0] == rightmost_enemy_x)] = abs(directional_distances[(bullets_on_side[:, 0] == rightmost_enemy_x)])

            setattr(self.state.bullet_state, distance_attr, np.array(sorted(directional_distances, key=abs)))
            non_zero_distances = getattr(self.state.bullet_state, distance_attr)
            non_zero_distances = non_zero_distances[non_zero_distances != 0]

            small_distances = [0.0] * StateConsts.PRACTICAL_MAX_NUM_ENEMY_BULLETS_PER_SIDE
            for i in range(min(len(non_zero_distances), StateConsts.PRACTICAL_MAX_NUM_ENEMY_BULLETS_PER_SIDE)):
                small_distances[i] = non_zero_distances[i]
            setattr(self.state.bullet_state, small_distance_attr, small_distances)

        update_player_bullet_distances(
            player=Players.HUMAN,
            bullet_side="left",
            distance_attr="human_distance_to_bullets",
            small_distance_attr="small_distance_to_bullets",
            leftmost_enemy_x=min(InitialStateConsts.HUMAN_ENEMIES_X),
            rightmost_enemy_x=max(InitialStateConsts.HUMAN_ENEMIES_X),
        )
        update_player_bullet_distances(
            player=Players.SHUTTER,
            bullet_side="right",
            distance_attr="shutter_distance_to_bullets",
            small_distance_attr="shutter_small_distance_to_bullets",
            leftmost_enemy_x=min(InitialStateConsts.SHUTTER_ENEMIES_X),
            rightmost_enemy_x=max(InitialStateConsts.SHUTTER_ENEMIES_X),
        )


    def check_bullet_collision(self, bullet1, bullet2, collision_distance=0.01):
        raise NotImplementedError("This function is deprecated. Handled in check_collisions(). Could migrate new implementation back here.")
        """
        Checks to see if two bullets have collided with one another
        """
        # Calculate the distance between two bullets.
        #TO DO: INCLUDE THIS IN CHECK COLLISIONS FUNCTION
        distance = ((bullet1[0] - bullet2[0]) ** 2 + (bullet1[1] - bullet2[1]) ** 2) ** 0.5
        return distance < collision_distance
    
    # TODO: share code for creating player/enemy rects and bullet rects
    def check_collisions(self,agent_powerup):
        """
        Checks to see if a bullets has collided with an agent's hitbox or if a bullet has collided with an enemy spaceship's hitbox
        """
        #bullet_size = 0.01  # Assuming this is an approximation of bullet dimensions relative to screen size
        enemy_width = DynamicsConsts.SHIP_WIDTH #50 / RenderConsts.SCREEN_WIDTH  # Enemy width adjusted for screen size
        enemy_height = DynamicsConsts.SHIP_HEIGHT #48 / RenderConsts.SCREEN_HEIGHT  # Enemy height adjusted for screen size
        bullets_to_remove = []
        enemy_bullets_to_remove = []

        previous_leading_player = self.state.get_leading_score_player(threshold=RewardConsts.LEADING_SCORE_THRESHOLD)

        # Check collisions between player bullets and enemies
        for i, (x, y) in enumerate(zip(InitialStateConsts.ENEMIES_X, InitialStateConsts.ENEMIES_Y)):
            if not self.state.enemy_state.enemies_active[i]:
                continue  # Skip inactive enemies

            # Create a rectangle for the enemy hitbox
            enemy_rect = pygame.Rect(
                (x - enemy_width / 2) * RenderConsts.SCREEN_WIDTH, 
                (y - enemy_height / 2) * RenderConsts.SCREEN_HEIGHT,
                enemy_width * RenderConsts.SCREEN_WIDTH,
                enemy_height * RenderConsts.SCREEN_HEIGHT
            )

            for index in self.state.bullet_state.playerb_used:
                bullet_x, bullet_y, shooter = self.state.bullet_state.player_bullets[index]
                # Create a rectangle for the bullet hitbox
                # TODO: what's 2 and 6.5?
                bullet_rect = pygame.Rect(
                    (bullet_x * RenderConsts.SCREEN_WIDTH)- 2, #checked using rectangle in render. Getting the top left and right points
                    (bullet_y  * RenderConsts.SCREEN_HEIGHT) -6.5,#checked using rectangle in render. 
                    RenderConsts.BULLET_WIDTH, #5, #width
                    RenderConsts.BULLET_HEIGHT #15 #length
                )

                # Check if the bullet rectangle collides with the enemy rectangle
                #TODO: Should StepInfo score changes track on the Nao's contribution, or is it fine as currently written?Nost
                if enemy_rect.colliderect(bullet_rect):
                    self.state.enemy_state.enemies_active[i] = False
                    score_bonus = ScoreConsts.BONUS_FOR_HITTING_ENEMY # 10
                    #TODO: make sure the score changes in the info dictionary properly reflects nao's contribution to human and/or shutter score 
                    if shooter == "Human":
                        #only count rewards if its from human
                        self.state.score_state.scores['Human'] += score_bonus
                        self.step_info.score_changes[Players.HUMAN] += score_bonus
                        self.step_info.enemy_elimination_rewards[Players(shooter)] += RewardConsts.ENEMY_ELIMINATION_REWARD
                    elif shooter == "Nao":
                        if x < 0.5:
                            # self.state.score_state.scores['Human'] += score_bonus #10  # Add score to Human if Nao shot it on the left side
                            self.state.score_state.scores['NaoForHuman'] += score_bonus
                            self.step_info.score_changes[Players.HUMAN] += score_bonus
                        else:
                            # self.state.score_state.scores['Shutter'] += score_bonus  # Add score to Shutter if Nao shot it on the right side
                            self.state.score_state.scores['NaoForShutter'] += score_bonus 
                            self.step_info.score_changes[Players.SHUTTER] += score_bonus
                    elif shooter == "Shutter":
                        self.state.score_state.scores['Shutter'] += score_bonus
                        self.step_info.score_changes[Players.SHUTTER] += score_bonus
                        self.step_info.enemy_elimination_rewards[Players(shooter)] += RewardConsts.ENEMY_ELIMINATION_REWARD
                    else:
                        raise ValueError(f"Invalid shooter: {shooter}")
                    
                    # TODO: should this double bonus stack with nao's assistance?
                    # self.state.score_state.scores[shooter] += score_bonus
                    # self.step_info.score_changes[Players(shooter)] += score_bonus
                    self.state.enemy_state.enemies_hit_frames[i] = 0

                    # Decrement multi bullets. If none left, mark for removal
                    self.state.bullet_state.player_multi_bullet_counts[index] -= 1
                    if self.state.bullet_state.player_multi_bullet_counts[index] == 0:
                        bullets_to_remove.append(index)  # Mark bullet for removal

        # Check collisions between enemy bullets and player bullets
        for index in self.state.bullet_state.enemyb_used:
            bullet_x, bullet_y = self.state.bullet_state.enemy_bullets[index]
            # Create a rectangle for the enemy bullet hitbox
            bullet_rect = pygame.Rect(
                (bullet_x * RenderConsts.SCREEN_WIDTH)- 2, #checked using rectangle in render. Getting the top left and right points
                (bullet_y  * RenderConsts.SCREEN_HEIGHT) -6.5,#checked using rectangle in render. 
                RenderConsts.BULLET_WIDTH, #5, #width
                RenderConsts.BULLET_HEIGHT #15 #length
            )

            for player_index in self.state.bullet_state.playerb_used:
                player_bullet_x, player_bullet_y, shooter = self.state.bullet_state.player_bullets[player_index]
                # Create a rectangle for the player bullet hitbox
                player_bullet_rect = pygame.Rect(
                    (player_bullet_x * RenderConsts.SCREEN_WIDTH)- 2,
                    (player_bullet_y  * RenderConsts.SCREEN_HEIGHT) -6.5,
                    RenderConsts.BULLET_WIDTH, #5, #width
                    RenderConsts.BULLET_HEIGHT #15 #length
                )

                if player_bullet_rect.colliderect(bullet_rect):
                    # Remove enemy bullet from the list
                    enemy_bullets_to_remove.append(index)

                    # Decrement player multi bullets if colliding with enemy bullets. If none left, mark for removal
                    self.state.bullet_state.player_multi_bullet_counts[player_index] -= 1
                    if self.state.bullet_state.player_multi_bullet_counts[player_index] == 0:
                        # Remove the player bullet from the list
                        bullets_to_remove.append(player_index)

        # Check collisions between enemy bullets and players (Human, Shutter, Nao)
        # TODO? nao getting hit during powerups? or should that not be possible?
        for index in self.state.bullet_state.enemyb_used:
            bullet_x, bullet_y = self.state.bullet_state.enemy_bullets[index]
            # TODO: what's 2 and 6.5?
            bullet_rect = pygame.Rect(
                (bullet_x * RenderConsts.SCREEN_WIDTH)- 2, #checked using rectangle in render. Getting the top left and right points
                (bullet_y  * RenderConsts.SCREEN_HEIGHT) -6.5,#checked using rectangle in render. 
                RenderConsts.BULLET_WIDTH, #5, #width
                RenderConsts.BULLET_HEIGHT #15 #length
            )

            # Define hit boxes for Human, Shutter, Nao
            #If Shutter is invincible, then Human hitbox must be active and deactivate Shutter hitbox
            if agent_powerup['Shutter'] == True:
                #print('Shutter is Invnicible')
                for player_name, player_x, player_y, player_active in [
                    ('Human', self.state.players_state.human_position_x, self.state.players_state.human_position_y, self.state.players_state.active['Human']),
                ]:
                    if player_name == 'Human':
                        #If Shutter has powerup only activate Human hitbox
                        player_rect = pygame.Rect(
                            (player_x - enemy_width / 2) * RenderConsts.SCREEN_WIDTH, 
                            (player_y - enemy_height / 2) * RenderConsts.SCREEN_HEIGHT,
                            enemy_width * RenderConsts.SCREEN_WIDTH,
                            enemy_height * RenderConsts.SCREEN_HEIGHT
                        )
                        # Check if any enemy bullet hits Human, Shutter, or Nao
                        if player_rect.colliderect(bullet_rect):
                            self.state.players_state.active['Human'] = False
                            self.state.history.last_hit_frame[player_name] = self.state.time_state.frame
                            self.step_info.players_hit.append(Players('Human'))
                            if player_name == 'Human':
                                self.step_info.player_hit_rewards[Players(player_name)] += RewardConsts.PLAYER_HIT_REWARD
                                # self.state.score_state.human_hit_count +=1
                            enemy_bullets_to_remove.append(index)

            #If Human is invincible, then Shutter hitbox must be active and deactivate Shutter hitbox
            elif agent_powerup['Human'] == True:
                #print('human is invincible')

                for player_name, player_x, player_y, player_active in [
                    ('Shutter', self.state.players_state.human_position_x, self.state.players_state.human_position_y, self.state.players_state.active['Human']),
                ]:
                    if player_name == 'Shutter': 
                        player_rect = pygame.Rect(
                            (player_x - enemy_width / 2) * RenderConsts.SCREEN_WIDTH, 
                            (player_y - enemy_height / 2) * RenderConsts.SCREEN_HEIGHT,
                            enemy_width * RenderConsts.SCREEN_WIDTH,
                            enemy_height * RenderConsts.SCREEN_HEIGHT
                        )
                        # Check if any enemy bullet hits Human, Shutter, or Nao
                        if player_rect.colliderect(bullet_rect):
                            self.state.players_state.active['Shutter'] = False
                            self.state.history.last_hit_frame[player_name] = self.state.time_state.frame
                            self.step_info.players_hit.append(Players('Shutter'))
                            if player_name == 'Shutter':
                                self.step_info.player_hit_rewards[Players(player_name)] += RewardConsts.PLAYER_HIT_REWARD
                                # self.state.score_state.shutter_hit_count +=1

                            enemy_bullets_to_remove.append(index)

            else:
                # print('neither is invincible')
                #In cases where the Nao is moving between players. If the powerup is not active both players have hitboxes
                # TODO: This is still causing both players to die at the same time whenever one of them gets hit.
                for player_name, player_x, player_y, player_active in [
                    ('Shutter', self.state.players_state.shutter_position_x, self.state.players_state.shutter_position_y, self.state.players_state.active['Shutter']),
                    ('Human', self.state.players_state.human_position_x, self.state.players_state.human_position_y, self.state.players_state.active['Human']),
                    ('Nao', self.state.players_state.nao_position_x, self.state.players_state.nao_position_y, self.state.players_state.active['Nao']),
                ]:
                    # Don't check for bullet collisions if the player is already knocked inactive
                    if not player_active:
                        continue

                    # If Nao is invincible, skip checking collisions with bullets
                    if PolicyConsts.NAO_INVINCIBLE and player_name == 'Nao':
                        continue

                    player_rect = pygame.Rect(
                        (player_x - enemy_width / 2) * RenderConsts.SCREEN_WIDTH, 
                        (player_y - enemy_height / 2) * RenderConsts.SCREEN_HEIGHT,
                        enemy_width * RenderConsts.SCREEN_WIDTH,
                        enemy_height * RenderConsts.SCREEN_HEIGHT
                    )
                    # Check if any enemy bullet hits Human, Shutter, or Nao
                    if player_rect.colliderect(bullet_rect):
                        self.state.players_state.active[player_name] = False
                        self.state.history.last_hit_frame[player_name] = self.state.time_state.frame
                        self.step_info.players_hit.append(Players(player_name))
                        if player_name == 'Human':
                            self.step_info.player_hit_rewards[Players(player_name)] += RewardConsts.PLAYER_HIT_REWARD
                            # self.state.score_state.human_hit_count +=1
                        elif player_name == 'Shutter':
                            self.step_info.player_hit_rewards[Players(player_name)] += RewardConsts.PLAYER_HIT_REWARD
                            # self.state.score_state.shutter_hit_count +=1
                        elif player_name == 'Nao':
                            self.step_info.player_hit_rewards[Players(player_name)] += RewardConsts.PLAYER_HIT_REWARD
                            # self.state.score_state.nao_hit_count +=1

                        enemy_bullets_to_remove.append(index)

        # Check score changes
        current_leading_player = self.state.get_leading_score_player(threshold=RewardConsts.LEADING_SCORE_THRESHOLD)
        if previous_leading_player != current_leading_player:
            if current_leading_player == Players.HUMAN:
                self.step_info.player_taking_score_lead_rewards[Players.HUMAN] += RewardConsts.PLAYER_TAKING_SCORE_LEAD_REWARD
                self.step_info.player_taking_score_lead_rewards[Players.SHUTTER] -= RewardConsts.PLAYER_TAKING_SCORE_LEAD_REWARD
            elif current_leading_player == Players.SHUTTER:
                self.step_info.player_taking_score_lead_rewards[Players.SHUTTER] += RewardConsts.PLAYER_TAKING_SCORE_LEAD_REWARD
                self.step_info.player_taking_score_lead_rewards[Players.HUMAN] -= RewardConsts.PLAYER_TAKING_SCORE_LEAD_REWARD
            elif current_leading_player == None:
                if previous_leading_player == Players.HUMAN:
                    self.step_info.player_taking_score_lead_rewards[Players.HUMAN] -= RewardConsts.PLAYER_TAKING_SCORE_LEAD_REWARD
                    self.step_info.player_taking_score_lead_rewards[Players.SHUTTER] += RewardConsts.PLAYER_TAKING_SCORE_LEAD_REWARD
                elif previous_leading_player == Players.SHUTTER:
                    self.step_info.player_taking_score_lead_rewards[Players.SHUTTER] -= RewardConsts.PLAYER_TAKING_SCORE_LEAD_REWARD
                    self.step_info.player_taking_score_lead_rewards[Players.HUMAN] += RewardConsts.PLAYER_TAKING_SCORE_LEAD_REWARD
                else:
                    raise ValueError(f"Previous leading player {previous_leading_player} not recognized")
            else:
                raise ValueError(f"Current leading player {current_leading_player} not recognized")
        
        # Remove bullets that hit an enemy
        for index in sorted(bullets_to_remove, reverse=True):
            self.state.bullet_state.player_bullets[index] = [0, 0, None]
            if index in self.state.bullet_state.playerb_used:
                self.state.bullet_state.playerb_used.remove(index)
            self.state.bullet_state.playerb_free.append(index)

        # Remove enemy bullets that hit players
        # NOTE: if multiple players collided with the bullet at the same time, it'll appear in the list multiple times, so set()


        for index in sorted(set(enemy_bullets_to_remove), reverse=True):
            bullet_x = self.state.bullet_state.enemy_bullets[index][0]
            bullet_y = self.state.bullet_state.enemy_bullets[index][1]
            
            # remove bullet from bullet_y_positions
            # TODO: update enemy bullets to include enemy idx so we don't need to reconstruct it to get the column index
            col_idx = InitialStateConsts.ENEMY_X_TO_COL_IDX[bullet_x]
            for slot in range(StateConsts.MAX_NUM_ENEMY_BULLETS_PER_COLUMN):
                if self.state.bullet_state.bullet_y_positions[col_idx, slot] == bullet_y:
                    self.state.bullet_state.bullet_y_positions[col_idx, slot] = 0
                    break
            
            # remove bullet from enemy_bullets
            self.state.bullet_state.enemy_bullets[index] = [0, 0]
            
            # mark the bullet index as free
            if index in self.state.bullet_state.enemyb_used:
                self.state.bullet_state.enemyb_used.remove(index)
            self.state.bullet_state.enemyb_free.append(index)
        
    # NOTE: currently, enemies in the same formation location respawn after a fixed time interval.
        # There's no end to the enemies, so they respawn indefinitely?
        # We may want some more visible sense of progression beyond just score.
            # e.g. winning the game by eliminating all enemies (or timing out if taking too long)
            # or if constant game time is important, respawn the entire formation once eliminated and winner by score
                # this would make a more visible marker of how many enemies are getting eliminated. Maybe some visible
                # marker on screen to show how many waves have been eliminated.
    def enemy_respawn_handler(self):
        """
        Respawn enemies after being eliminated
        """

        if DynamicsConsts.ENEMY_WAIT_FOR_HALF_ELIMINATION_TO_RESPAWN:
            # Get active enemies by side
            active_enemies_left, active_enemies_right = self.state.get_active_enemies_by_side()

            # Check if all enemies on the left half are eliminated
            all_left_eliminated = len(active_enemies_left) == 0

            # Check if all enemies on the right half are eliminated
            all_right_eliminated = len(active_enemies_right) == 0

            # Handle respawn for the left half
            if all_left_eliminated:
                for i, x in enumerate(InitialStateConsts.ENEMIES_X):
                    if x < 0.5:  # Left half of the screen
                        if self.state.enemy_state.enemies_hit_frames[i] >= 0:  # Enemy has been hit
                            self.state.enemy_state.enemies_hit_frames[i] += 1
                            if self.state.enemy_state.enemies_hit_frames[i] >= DynamicsConsts.RESPAWN_FREQUENCY:
                                self.state.enemy_state.enemies_active[i] = True
                                self.state.enemy_state.enemies_hit_frames[i] = -1  # Reset frame counter

            # Handle respawn for the right half
            if all_right_eliminated:
                for i, x in enumerate(InitialStateConsts.ENEMIES_X):
                    if x >= 0.5:  # Right half of the screen
                        if self.state.enemy_state.enemies_hit_frames[i] >= 0:  # Enemy has been hit
                            self.state.enemy_state.enemies_hit_frames[i] += 1
                            if self.state.enemy_state.enemies_hit_frames[i] >= DynamicsConsts.RESPAWN_FREQUENCY:
                                self.state.enemy_state.enemies_active[i] = True
                                self.state.enemy_state.enemies_hit_frames[i] = -1  # Reset frame counter
        else:
            #print('enemies hit frames', self.state.enemy_state.enemies_hit_frames)
            for i in range(len(self.state.enemy_state.enemies_hit_frames)):
                if self.state.enemy_state.enemies_hit_frames[i] >= 0:  # Enemy has been hit
                    self.state.enemy_state.enemies_hit_frames[i] += 1
                    if self.state.enemy_state.enemies_hit_frames[i] >= DynamicsConsts.RESPAWN_FREQUENCY: #300 :  # Respawn enemy after 50 frames
                        self.state.enemy_state.enemies_active[i] = True
                        self.state.enemy_state.enemies_hit_frames[i] = -1  # Reset frame counter

    # TODO? propagate gamepad controller objects to the environment?
    def wait_for_cue(self, continue_cue: str):
        if self.config.interactive_mode == InteractiveModes.NONE:
            # No interactive mode, so wait for brief fixed time instead
            time.sleep(DynamicsConsts.NONINTERACTIVE_WAIT_TIME_SECONDS)
            return []
        
        if continue_cue not in ["keyboard_space", "gamepad_a_button", "all_gamepad_a_button"]:
            raise ValueError(f"Unsupported continue cue: {continue_cue}. Supported cues are 'keyboard_space' and 'gamepad_a_button' and 'all_gamepad_a_button'.")
        if continue_cue in ["gamepad_a_button", "all_gamepad_a_button"]:
            assert self.config.interactive_mode == InteractiveModes.GAMEPAD, "Gamepad continue cue can only be used in gamepad interactive mode."

        # Wait for the continue cue
        if continue_cue == "keyboard_space":
            # Wait for space key press
            cue_events = [PygameEventWaiter.wait_for_keyboard_press(key="space")]
        elif continue_cue == "gamepad_a_button":
            # Wait for A button press on any gamepad
            cue_events = [PygameEventWaiter.wait_for_gamepad_press(button_idx=PyGameGamepad.XboxButtons.A.value)]
        elif continue_cue == "all_gamepad_a_button":
            # Wait for A button press on all gamepads
            cue_events = PygameEventWaiter.wait_for_all_gamepads_press(button_idx=PyGameGamepad.XboxButtons.A.value)

        # Clear the event queue to discard any events after the cue
        pygame.event.clear()

        return cue_events

    def display_blocking_dialog_message(self, message: str, continue_cue="keyboard_space", show_continue_cue=True, countdown_seconds: int = 0, forced_wait_time_seconds: int = 5):
        if self.screen is None:
            # TODO: self.screen shouldn't be None even in headless mode, so this case shouldn't happen.
            import pdb; pdb.set_trace()
            return
        
        DIALOG_WIDTH = 0.7
        DIALOG_HEIGHT = 0.65
        DIALOG_COLOR = (0, 0, 0)  # Black background
        BORDER_COLOR = (255, 255, 255)  # White border
        TEXT_COLOR = (255, 255, 255)  # White text
        DARK_PINK_COLOR = (200, 100, 150)  # Darker pink for human's contribution
        DARK_TURQUOISE_COLOR = (0, 200, 150)  # Darker turquoise for shutter's contribution
        PINK_COLOR = (255, 105, 180)  # Pink for human's contribution
        TURQUOISE_COLOR = (64, 224, 208)  # Turquoise for shutter's contribution

        if self.config.interactive_mode == InteractiveModes.KEYBOARD:
            print("Defaulting to keyboard space continue cue since gamepad interactive mode is not enabled.")
            continue_cue = "keyboard_space"

        # Track rendering of each frame during dialog display
        rendered_frames = []
        
        # Clear the screen with a semi-transparent overlay
        self.screen.fill((0, 0, 0))  # Clear the screen to remove the old dialog box
        self.render() # Add back the game objects

        # Calculate dialog box dimensions
        dialog_width = DIALOG_WIDTH * RenderConsts.SCREEN_WIDTH*RenderConsts.RENDER_UPSCALE_FACTOR
        dialog_height = DIALOG_HEIGHT * RenderConsts.SCREEN_HEIGHT*RenderConsts.RENDER_UPSCALE_FACTOR

        # Calculate dialog box position (centered on the screen)
        dialog_x = (RenderConsts.SCREEN_WIDTH*RenderConsts.RENDER_UPSCALE_FACTOR - dialog_width) // 2
        dialog_y = (RenderConsts.SCREEN_HEIGHT*RenderConsts.RENDER_UPSCALE_FACTOR - dialog_height) // 2

        # Create a semi-transparent surface for the dialog box
        dialog_surface = pygame.Surface((dialog_width, dialog_height), pygame.SRCALPHA)
        dialog_surface.fill((0, 0, 0, 200))  # Black with 50% transparency (alpha = 128)
        # Blit the semi-transparent surface onto the screen
        self.screen.blit(dialog_surface, (dialog_x, dialog_y))
        # Draw the white border
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height), 3)  # White border

        # Render the message text
        font = self.rendered_objects.font

        # Render the message text (wrapped)
        max_text_width = dialog_width - 40  # Leave some padding inside the dialog box
        lines = []
        words = message.split(' ')
        current_line = ""

        # Split the message into lines based on \n
        raw_lines = message.split('\n')

        for raw_line in raw_lines:
            # Determine the color for the entire raw line
            if "left" in raw_line.lower() and "right" in raw_line.lower():
                line_color = TEXT_COLOR  # Default color
            elif "left" in raw_line.lower():
                line_color = PINK_COLOR  # Color for "left"
            elif "right" in raw_line.lower():
                line_color = TURQUOISE_COLOR  # Color for "right"
            else:
                line_color = TEXT_COLOR  # Default color

            words = raw_line.split(' ')
            current_line = ""

            for word in words:
                # Check if adding the next word exceeds the max width
                test_line = f"{current_line} {word}".strip()
                if font.size(test_line)[0] <= max_text_width:
                    current_line = test_line
                else:
                    lines.append((current_line, line_color))
                    current_line = word

            # Add the last line of the current raw line
            if current_line:
                lines.append((current_line, line_color))

            # Add an empty line after each raw line (to create spacing for \n)
            lines.append(("", TEXT_COLOR))  # Empty line for spacing

        # Remove the last empty line if it exists
        if lines and lines[-1] == "":
            lines.pop()

        # if show_continue_cue:
        #     lines.append(("", TEXT_COLOR))  # Add an empty line for spacing before the continue cue
        #     lines.append(("", TEXT_COLOR))  # Add another empty line for spacing
        #     if continue_cue == "keyboard_space":
        #         lines.append(("Press SPACE to continue", TEXT_COLOR))
        #     elif continue_cue == "gamepad_a_button":
        #         lines.append(("Press A on the gamepad to continue", TEXT_COLOR))
        #     elif continue_cue == "all_gamepad_a_button":
        #         lines.append(("Press A on both gamepads to continue", TEXT_COLOR))
        #     else:
        #         raise ValueError(f"Unsupported continue cue: {continue_cue}. Supported cues are 'keyboard_space', 'gamepad_a_button', and 'all_gamepad_a_button'.")

        # Render each line of text
        line_height = font.size("Tg")[1]  # Height of a single line of text
        total_text_height = len(lines) * line_height
        start_y = dialog_y + (dialog_height - total_text_height) // 2  # Center the text vertically

        for i, (line, line_color) in enumerate(lines):
            text_surface = font.render(line, True, line_color)
            text_rect = text_surface.get_rect(center=(dialog_x + dialog_width // 2, start_y + i * line_height))
            self.screen.blit(text_surface, text_rect)


        # Update the display
        if self._using_visible_display():
            pygame.display.update()
        rendered_frames.append(self._get_display_as_array())

        # Force waiting a few seconds before allowing advancement past dialog box
        pygame.time.delay(forced_wait_time_seconds * 1000)  # Delay for 5 seconds

        # Clear the event queue to discard any events during the delay
        #if self._using_visible_display():
        if self.config.interactive_mode != InteractiveModes.NONE:
            # NOTE: pygame event cue is not available in non-interactive mode
            pygame.event.clear()

        # Add the continue cue message after force wait time (or immediately if no force wait time)
        if show_continue_cue:
            lines.append(("", TEXT_COLOR))  # Add an empty line for spacing before the continue cue
            lines.append(("", TEXT_COLOR))  # Add another empty line for spacing
            if continue_cue == "keyboard_space":
                lines.append(("Press SPACE to continue", TEXT_COLOR))
            elif continue_cue == "gamepad_a_button":
                lines.append(("Press A on the gamepad to continue", TEXT_COLOR))
            elif continue_cue == "all_gamepad_a_button":
                lines.append(("Press A on both gamepads to continue", TEXT_COLOR))
            else:
                raise ValueError(f"Unsupported continue cue: {continue_cue}. Supported cues are 'keyboard_space', 'gamepad_a_button', and 'all_gamepad_a_button'.")

            # Re-render the dialog box with the continue cue
            self.screen.fill((0, 0, 0))  # Clear the screen
            self.render()  # Add back the game objects
            self.screen.blit(dialog_surface, (dialog_x, dialog_y))
            pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height), 3)  # White border

            # Re-render the message text
            for i, (line, line_color) in enumerate(lines):
                text_surface = font.render(line, True, line_color)
                text_rect = text_surface.get_rect(center=(dialog_x + dialog_width // 2, start_y + i * line_height))
                self.screen.blit(text_surface, text_rect)

            # Update the display
            if self._using_visible_display():
                pygame.display.update()
            rendered_frames.append(self._get_display_as_array())

        # # Wait for user input to close the dialog
        cue_events = self.wait_for_cue(continue_cue)
        
        # Optional countdown before resuming the game
        if countdown_seconds > 0:
            font = pygame.font.Font(None, 72)  # Larger font for countdown
            for count in ["5", "4", "3", "2", "1", "Go!"]:
                # Clear the screen with a semi-transparent overlay
                self.screen.fill((0, 0, 0))  # Clear the screen to remove the old dialog box
                self.render() # Add back the game objects

                ####
                # Create a semi-transparent surface for the dialog box
                # dialog_surface = pygame.Surface((dialog_width, dialog_height), pygame.SRCALPHA)
                # dialog_surface.fill((0, 0, 0, 64))  # Black with 25% transparency (alpha = 64)

                # # Blit the semi-transparent surface onto the screen
                # self.screen.blit(dialog_surface, (dialog_x, dialog_y))

                # # Draw the white border
                # pygame.draw.rect(self.screen, (255, 255, 255),
                # pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height), 3)  # White border
                #####

                # # Render the countdown text
                text_surface = font.render(count, True, (255, 255, 255))  # White text
                text_rect = text_surface.get_rect(center=(dialog_x + dialog_width // 2, dialog_y + dialog_height // 2))
                self.screen.blit(text_surface, text_rect)

                # Update the display
                if self._using_visible_display():
                    pygame.display.update()
                rendered_frames.append(self._get_display_as_array())

                # Wait for 1 second before showing the next number
                pygame.time.delay(1000)
        
        return cue_events, rendered_frames
    
    def render_scores_old(self):
        human_score = int(self.state.score_state.scores['Human'])
        shutter_score = int(self.state.score_state.scores['Shutter'])
        nao_score_for_human = int(self.state.score_state.scores['NaoForHuman'])
        nao_score_for_shutter = int(self.state.score_state.scores['NaoForShutter'])

        PINK_COLOR = (255, 192, 203)    # Pink for human score
        TURQUOISE_COLOR = (0, 255, 200) # Turquoise for shutter score
        WHITE_COLOR = (255, 255, 255)   # White for NAO's contributions

        # render player 1 score components
        p1_score_text = f"YOUR SCORE: {human_score + nao_score_for_human} = {human_score}"
        p1_nao_text = f" + {nao_score_for_human}"
        
        # get total width for player 1 score for centering 
        p1_score_width = self.rendered_objects.font.size(p1_score_text)[0]
        p1_nao_width = self.rendered_objects.font.size(p1_nao_text)[0]
        p1_total_width = p1_score_width + p1_nao_width
        
        # starting x position for perfect centering
        p1_start_x = (RenderConsts.SCREEN_WIDTH - p1_total_width) / 2
        
        # human contribution in pink
        p1_score_surface = self.rendered_objects.font.render(p1_score_text, True, PINK_COLOR)
        p1_score_rect = p1_score_surface.get_rect()
        p1_score_rect.left = p1_start_x
        p1_score_rect.centery = RenderConsts.SCORE_BOX_PLAYER1_VERTICAL_OFFSET
        self.screen.blit(p1_score_surface, p1_score_rect)
        
        # nao's contribution in white 
        p1_nao_surface = self.rendered_objects.font.render(p1_nao_text, True, WHITE_COLOR)
        p1_nao_rect = p1_nao_surface.get_rect()
        p1_nao_rect.left = p1_score_rect.right
        p1_nao_rect.centery = p1_score_rect.centery
        self.screen.blit(p1_nao_surface, p1_nao_rect)

        # same thing for p2 
        p2_score_text = f"SHUTTER SCORE: {shutter_score + nao_score_for_shutter} = {shutter_score}"
        p2_nao_text = f" + {nao_score_for_shutter}"
        
        p2_score_width = self.rendered_objects.font.size(p2_score_text)[0]
        p2_nao_width = self.rendered_objects.font.size(p2_nao_text)[0]
        p2_total_width = p2_score_width + p2_nao_width
        
        p2_start_x = (RenderConsts.SCREEN_WIDTH - p2_total_width) / 2
        
        p2_score_surface = self.rendered_objects.font.render(p2_score_text, True, TURQUOISE_COLOR)
        p2_score_rect = p2_score_surface.get_rect()
        p2_score_rect.left = p2_start_x
        p2_score_rect.centery = RenderConsts.SCORE_BOX_PLAYER2_VERTICAL_OFFSET
        self.screen.blit(p2_score_surface, p2_score_rect)
        
        p2_nao_surface = self.rendered_objects.font.render(p2_nao_text, True, WHITE_COLOR)
        p2_nao_rect = p2_nao_surface.get_rect()
        p2_nao_rect.left = p2_score_rect.right
        p2_nao_rect.centery = p2_score_rect.centery
        self.screen.blit(p2_nao_surface, p2_nao_rect)

    def render_scores_split_bar(self):
        human_score = int(self.state.score_state.scores['Human'])
        shutter_score = int(self.state.score_state.scores['Shutter'])
        nao_score_for_human = int(self.state.score_state.scores['NaoForHuman'])
        nao_score_for_shutter = int(self.state.score_state.scores['NaoForShutter'])

        # Colors
        DARK_PINK_COLOR = (200, 100, 150)  # Darker pink for human's contribution
        DARK_TURQUOISE_COLOR = (0, 200, 150)  # Darker turquoise for shutter's contribution
        WHITE_COLOR = (255, 255, 255)  # White for Nao's contributions
        GRAY_COLOR = (200, 200, 200)  # Background bar color
        VICTORY_YELLOW_COLOR = (255, 215, 0)  # Victory yellow color for the progress bar

        # Total scores
        total_human_score = human_score + nao_score_for_human
        total_shutter_score = shutter_score + nao_score_for_shutter
        score_per_enemy_formation = ScoreConsts.BONUS_FOR_HITTING_ENEMY * StateConsts.NUM_ENEMIES / StateConsts.NUM_SIDES
        if self.config.independent_victory_score_threshold is None:
            max_total_score = score_per_enemy_formation
            while max(total_human_score, total_shutter_score) > max_total_score:
                max_total_score += score_per_enemy_formation
        else:
            max_total_score = self.config.independent_victory_score_threshold * RenderConsts.MAX_BAR_SIZE_VS_VICTORY_THRESHOLD_SCALE_FACTOR

        # Bar dimensions
        total_bar_width = (RenderConsts.SCREEN_WIDTH - RenderConsts.PROGRESS_BAR_START_X - RenderConsts.TEXT_PADDING)
        bar_width_human = (total_human_score / max_total_score) * total_bar_width
        bar_width_shutter = (total_shutter_score / max_total_score) * total_bar_width

        # Calculate sub-bar widths
        human_contribution_width = (human_score / max_total_score) * total_bar_width
        nao_for_human_width = (nao_score_for_human / max_total_score) * total_bar_width
        shutter_contribution_width = (shutter_score / max_total_score) * total_bar_width
        nao_for_shutter_width = (nao_score_for_shutter / max_total_score) * total_bar_width

        # If scaling scores against a fixed threshold, once either player hits that threshold,
        # keep the bar at maximum screen width but adjust relative contributions of each player
        if self.config.independent_victory_score_threshold is not None and total_human_score > max_total_score:
            # Scale down contributions if total score exceeds max_total_score
            scale_factor = max_total_score / total_human_score
            bar_width_human *= scale_factor
            human_contribution_width *= scale_factor
            nao_for_human_width *= scale_factor
        if self.config.independent_victory_score_threshold is not None and total_shutter_score > max_total_score:
            # Scale down contributions if total score exceeds max_total_score
            scale_factor = max_total_score / total_shutter_score
            bar_width_shutter *= scale_factor
            shutter_contribution_width *= scale_factor
            nao_for_shutter_width *= scale_factor

        # Vertical positions
        human_bar_y = RenderConsts.SCORE_BOX_PLAYER1_VERTICAL_OFFSET
        shutter_bar_y = RenderConsts.SCORE_BOX_PLAYER2_VERTICAL_OFFSET

        # Render text for human score
        #human_score_text = f"{human_score:04d}"  # Player's score as a 4-digit string
        #human_nao_score_text = f"+{nao_score_for_human:04d}"  # Nao's contribution as a 4-digit string with a leading '+'
        human_score_text = f"{total_human_score:04d}"
        human_nao_score_text = ""

        # Render the player's score in DARK_PINK_COLOR
        human_score_surface = self.rendered_objects.font.render(human_score_text, True, DARK_PINK_COLOR)
        human_score_rect = human_score_surface.get_rect()
        human_score_rect.left = RenderConsts.TEXT_PADDING  # Fixed left alignment
        human_score_rect.centery = human_bar_y + RenderConsts.SCORE_BOX_HEIGHT // 2

        # Render Nao's contribution in WHITE_COLOR
        human_nao_score_surface = self.rendered_objects.font.render(human_nao_score_text, True, WHITE_COLOR)
        human_nao_score_rect = human_nao_score_surface.get_rect()
        human_nao_score_rect.left = human_score_rect.right  # Align Nao's text to the right of the player's score
        human_nao_score_rect.centery = human_score_rect.centery

        # Blit both parts of the text
        self.screen.blit(human_score_surface, human_score_rect)
        self.screen.blit(human_nao_score_surface, human_nao_score_rect)

        # Render text for shutter score
        #shutter_score_text = f"{shutter_score:04d}"  # Player's score as a 4-digit string
        #shutter_nao_score_text = f"+{nao_score_for_shutter:04d}"  # Nao's contribution as a 4-digit string with a leading '+'
        shutter_score_text = f"{total_shutter_score:04d}"
        shutter_nao_score_text = ""

        # Render the player's score in DARK_TURQUOISE_COLOR
        shutter_score_surface = self.rendered_objects.font.render(shutter_score_text, True, DARK_TURQUOISE_COLOR)
        shutter_score_rect = shutter_score_surface.get_rect()
        shutter_score_rect.left = RenderConsts.TEXT_PADDING  # Fixed left alignment
        shutter_score_rect.centery = shutter_bar_y + RenderConsts.SCORE_BOX_HEIGHT // 2

        # Render Nao's contribution in WHITE_COLOR
        shutter_nao_score_surface = self.rendered_objects.font.render(shutter_nao_score_text, True, WHITE_COLOR)
        shutter_nao_score_rect = shutter_nao_score_surface.get_rect()
        shutter_nao_score_rect.left = shutter_score_rect.right  # Align Nao's text to the right of the player's score
        shutter_nao_score_rect.centery = shutter_score_rect.centery

        # Blit both parts of the text
        self.screen.blit(shutter_score_surface, shutter_score_rect)
        self.screen.blit(shutter_nao_score_surface, shutter_nao_score_rect)

        
        # Render human score bar (stacked)
        # is_human_victory = (
        #     ScoreConsts.INDEPENDENT_VICTORY_THRESHOLD is not None
        #     and total_human_score >= max_total_score
        # )
        is_human_victory = self.state.get_player_independent_victory(player=Players.HUMAN, config=self.config)
        human_bar_background_color = VICTORY_YELLOW_COLOR if is_human_victory else GRAY_COLOR
        human_bar_rect = pygame.Rect(
            RenderConsts.PROGRESS_BAR_START_X,  # Start after the text
            human_bar_y,
            bar_width_human,
            RenderConsts.SCORE_BOX_HEIGHT
        )
        pygame.draw.rect(self.screen, human_bar_background_color, human_bar_rect)  # Background bar

        # Player's contribution (darker pink)
        human_contribution_rect = pygame.Rect(
            RenderConsts.PROGRESS_BAR_START_X,
            human_bar_y,
            human_contribution_width,
            RenderConsts.SCORE_BOX_HEIGHT
        )
        pygame.draw.rect(self.screen, DARK_PINK_COLOR, human_contribution_rect.inflate(-4, -4))  # Foreground bar

        # Nao's contribution (white)
        nao_for_human_rect = pygame.Rect(
            RenderConsts.PROGRESS_BAR_START_X + human_contribution_width,
            human_bar_y,
            nao_for_human_width,
            RenderConsts.SCORE_BOX_HEIGHT
        )
        pygame.draw.rect(self.screen, WHITE_COLOR, nao_for_human_rect.inflate(-4, -4))  # Foreground bar

        # Render shutter score bar (stacked)
        # is_shutter_victory = (
        #     ScoreConsts.INDEPENDENT_VICTORY_THRESHOLD is not None
        #     and total_shutter_score >= max_total_score
        # )
        is_shutter_victory = self.state.get_player_independent_victory(player=Players.SHUTTER, config=self.config)
        shutter_bar_background_color = VICTORY_YELLOW_COLOR if is_shutter_victory else GRAY_COLOR
        shutter_bar_rect = pygame.Rect(
            RenderConsts.PROGRESS_BAR_START_X,  # Start after the text
            shutter_bar_y,
            bar_width_shutter,
            RenderConsts.SCORE_BOX_HEIGHT
        )
        pygame.draw.rect(self.screen, shutter_bar_background_color, shutter_bar_rect)  # Background bar

        # Player's contribution (darker turquoise)
        shutter_contribution_rect = pygame.Rect(
            RenderConsts.PROGRESS_BAR_START_X,
            shutter_bar_y,
            shutter_contribution_width,
            RenderConsts.SCORE_BOX_HEIGHT
        )
        pygame.draw.rect(self.screen, DARK_TURQUOISE_COLOR, shutter_contribution_rect.inflate(-4, -4))  # Foreground bar

        # Nao's contribution (white)
        nao_for_shutter_rect = pygame.Rect(
            RenderConsts.PROGRESS_BAR_START_X + shutter_contribution_width,
            shutter_bar_y,
            nao_for_shutter_width,
            RenderConsts.SCORE_BOX_HEIGHT
        )
        pygame.draw.rect(self.screen, WHITE_COLOR, nao_for_shutter_rect.inflate(-4, -4))  # Foreground bar
        
    def render_scores(self):
        if self.config.independent_victory_score_threshold is not None:
            raise NotImplementedError("render_scores() does not support independent victory threshold yet. Use render_scores_split_bar() instead, or steal code from there to implement it here.")

        human_score = int(self.state.score_state.scores['Human'])
        shutter_score = int(self.state.score_state.scores['Shutter'])
        nao_score_for_human = int(self.state.score_state.scores['NaoForHuman'])
        nao_score_for_shutter = int(self.state.score_state.scores['NaoForShutter'])

        PINK_COLOR = (255, 192, 203)    # Pink for human score
        TURQUOISE_COLOR = (0, 255, 200) # Turquoise for shutter score
        WHITE_COLOR = (255, 255, 255)   # White for NAO's contributions
        GRAY_COLOR = (200, 200, 200)    # Background bar color

        shutter_bar_y = RenderConsts.SCORE_BOX_PLAYER2_VERTICAL_OFFSET

        # Total scores
        total_human_score = human_score + nao_score_for_human
        total_shutter_score = shutter_score + nao_score_for_shutter
        #total_score = max(1, total_human_score + total_shutter_score) # Total space for both bars
        #max_total_score = 1e-6 + max(1000, total_human_score, total_shutter_score) # Total space for both bars
        score_per_enemy_formation = ScoreConsts.BONUS_FOR_HITTING_ENEMY * StateConsts.NUM_ENEMIES / StateConsts.NUM_SIDES
        max_total_score = score_per_enemy_formation
        while max(total_human_score, total_shutter_score) > max_total_score:
            max_total_score += score_per_enemy_formation

        # Bar dimensions
        bar_width_human = (total_human_score / max_total_score) * (RenderConsts.SCREEN_WIDTH - RenderConsts.PROGRESS_BAR_START_X - RenderConsts.TEXT_PADDING)
        bar_width_shutter = (total_shutter_score / max_total_score) * (RenderConsts.SCREEN_WIDTH - RenderConsts.PROGRESS_BAR_START_X - RenderConsts.TEXT_PADDING)

        # Vertical positions
        human_bar_y = RenderConsts.SCORE_BOX_PLAYER1_VERTICAL_OFFSET
        shutter_bar_y = RenderConsts.SCORE_BOX_PLAYER2_VERTICAL_OFFSET

        # Render text for human score
        #human_text = "LEFT PLAYER"
        human_text = f"{total_human_score:04d}"  # Format the score as a 4-digit string, padded with zeros if necessary
        human_text_surface = self.rendered_objects.font.render(human_text, True, PINK_COLOR)
        human_text_rect = human_text_surface.get_rect()
        human_text_rect.left = RenderConsts.TEXT_PADDING  # Fixed left alignment
        human_text_rect.centery = human_bar_y + RenderConsts.SCORE_BOX_HEIGHT // 2
        self.screen.blit(human_text_surface, human_text_rect)

        # Render text for shutter score
        #shutter_text = "RIGHT PLAYER"
        shutter_text = f"{total_shutter_score:04d}"  # Format the score as a 4-digit string, padded with zeros if necessary
        shutter_text_surface = self.rendered_objects.font.render(shutter_text, True, TURQUOISE_COLOR)
        shutter_text_rect = shutter_text_surface.get_rect()
        shutter_text_rect.left = RenderConsts.TEXT_PADDING  # Fixed left alignment
        shutter_text_rect.centery = shutter_bar_y + RenderConsts.SCORE_BOX_HEIGHT // 2
        self.screen.blit(shutter_text_surface, shutter_text_rect)

        # Render human score bar
        human_bar_rect = pygame.Rect(
            RenderConsts.PROGRESS_BAR_START_X,  # Start after the text
            human_bar_y,
            bar_width_human,
            RenderConsts.SCORE_BOX_HEIGHT
        )
        pygame.draw.rect(self.screen, GRAY_COLOR, human_bar_rect)  # Background bar
        pygame.draw.rect(self.screen, PINK_COLOR, human_bar_rect.inflate(-4, -4))  # Foreground bar

        # Render shutter score bar
        shutter_bar_rect = pygame.Rect(
            RenderConsts.PROGRESS_BAR_START_X,  # Start after the text
            shutter_bar_y,
            bar_width_shutter,
            RenderConsts.SCORE_BOX_HEIGHT
        )
        pygame.draw.rect(self.screen, GRAY_COLOR, shutter_bar_rect)  # Background bar
        pygame.draw.rect(self.screen, TURQUOISE_COLOR, shutter_bar_rect.inflate(-4, -4))  # Foreground bar

    def render_time_progress_bar(self):
        """
        Render the time progress bar at the bottom of the screen
        """
        # TODO: properly define some of the constants for score bars and progress bars to make future attempts at resizing less brittle
            # particularly score bar height = 30 that was copied from render_scores()

        # Define colors
        WHITE_COLOR = (175, 175, 175)#(255, 255, 0)#(150, 150, 150)#(255, 255, 255)
        BLACK_COLOR = (0, 0, 0)

        # Position time progress bar
        time_bar_y = RenderConsts.SCORE_BOX_PLAYER2_VERTICAL_OFFSET + RenderConsts.SCORE_BOX_HEIGHT + 5  # Position below the score bars with some padding
        time_bar_width = RenderConsts.SCREEN_WIDTH - RenderConsts.PROGRESS_BAR_START_X - RenderConsts.TEXT_PADDING  # Full width minus padding
        time_bar_height = 20  # Height of the time progress bar

        # Render "TIME" label to the left of the time progress bar
        time_label_surface = self.rendered_objects.font.render("Time", True, WHITE_COLOR)  # Render the text
        time_label_rect = time_label_surface.get_rect()
        time_label_rect.left = RenderConsts.TEXT_PADDING  # Align with the left padding
        time_label_rect.centery = time_bar_y + time_bar_height // 2  # Center vertically with the time bar
        self.screen.blit(time_label_surface, time_label_rect)
        
        # Calculate the width of the filled portion of the time bar
        # NOTE: frame is 0-indexed, so duration - 1 is the last frame
        time_progress = self.state.time_state.frame / (self.config.game_duration_frames - 1)
        filled_time_bar_width = time_progress * time_bar_width

        # Draw the outline of the time bar
        time_bar_rect = pygame.Rect(
            RenderConsts.PROGRESS_BAR_START_X,
            time_bar_y,
            time_bar_width,
            time_bar_height
        )
        pygame.draw.rect(self.screen, WHITE_COLOR, time_bar_rect, 2)  # Outline with 2-pixel thickness

        # Draw the filled portion of the time bar
        filled_time_bar_rect = pygame.Rect(
            RenderConsts.PROGRESS_BAR_START_X,
            time_bar_y,
            filled_time_bar_width,
            time_bar_height
        )
        #pygame.draw.rect(self.screen, (255, 255, 0), filled_time_bar_rect)  # Yellow fill for current progress
        pygame.draw.rect(self.screen, WHITE_COLOR, filled_time_bar_rect)  # Yellow fill for current progress
        
        # Draw segment dividers
        num_segments = self.config.num_game_segments
        if num_segments <= 1:
            return

        segment_width = time_bar_width / num_segments  # Width of each segment
        for i in range(1, num_segments):  # Skip the first segment (start of the bar)
            segment_x = RenderConsts.PROGRESS_BAR_START_X + i * segment_width
            # pygame.draw.line(
            #     self.screen,
            #     WHITE_COLOR,  # Color of the segment divider
            #     (segment_x, time_bar_y),  # Start point of the line
            #     (segment_x, time_bar_y + time_bar_height),  # End point of the line
            #     2#1  # Line thickness
            # )
            line_color = WHITE_COLOR if i * segment_width > filled_time_bar_width else BLACK_COLOR
            pygame.draw.line(
                self.screen,
                line_color,  # Color of the segment divider
                (segment_x, time_bar_y),  # Start point of the line
                (segment_x, time_bar_y + time_bar_height),  # End point of the line
                2  # Line thickness
            )
    
    def render_enemy_progress_bar(self):
        ### Render progress bar for how many enemies have been eliminated
        
        if self.config.game_type == GameTypes.COMPETITIVE_TUTORIAL:
            # In competitive tutorial mode, render progress bar for the interactive player
            assert len(self.config.interactive_players) == 1, "Competitive tutorial mode should have exactly one interactive player."
            interactive_player = self.config.interactive_players[0]
            num_enemy_eliminations = self.state.get_cumulative_num_enemy_eliminations(
                player=interactive_player,
            )
            max_num_enemy_eliminations = DynamicsConsts.TUTORIAL_NUM_ENEMIES_TO_ELIMINATE
        else:
            raise ValueError(f"Unsupported game type {self.config.game_type} for rendering enemy progress bar.")

        # Define colors
        WHITE_COLOR = (255, 255, 255)
        BLACK_COLOR = (0, 0, 0)

        # Position enemy progress bar
        enemy_progress_bar_start_x = RenderConsts.PROGRESS_BAR_START_X + 75
        enemy_bar_y = RenderConsts.SCORE_BOX_PLAYER2_VERTICAL_OFFSET + RenderConsts.SCORE_BOX_HEIGHT + 5  # Position below the score bars with extra padding
        enemy_bar_width = RenderConsts.SCREEN_WIDTH - enemy_progress_bar_start_x - RenderConsts.TEXT_PADDING  # Full width minus padding
        enemy_bar_height = 20  # Height of the enemy progress bar

        # Render "Enemies" label to the left of the enemy progress bar
        enemy_label_surface = self.rendered_objects.font.render("Enemies", True, WHITE_COLOR)  # Render the text
        enemy_label_rect = enemy_label_surface.get_rect()
        enemy_label_rect.left = RenderConsts.TEXT_PADDING  # Align with the left padding
        enemy_label_rect.centery = enemy_bar_y + enemy_bar_height // 2  # Center vertically with the enemy bar
        self.screen.blit(enemy_label_surface, enemy_label_rect)

        # Calculate the width of the filled portion of the enemy bar
        num_enemy_eliminations = self.state.get_cumulative_num_enemy_eliminations(
            player=self.config.interactive_players[0],  # Assuming one interactive player
        )
        max_num_enemy_eliminations = DynamicsConsts.TUTORIAL_NUM_ENEMIES_TO_ELIMINATE
        enemy_progress = num_enemy_eliminations / max_num_enemy_eliminations
        filled_enemy_bar_width = enemy_progress * enemy_bar_width

        # Draw the outline of the enemy bar
        enemy_bar_rect = pygame.Rect(
            enemy_progress_bar_start_x,
            enemy_bar_y,
            enemy_bar_width,
            enemy_bar_height
        )
        pygame.draw.rect(self.screen, WHITE_COLOR, enemy_bar_rect, 2)  # Outline with 2-pixel thickness

        # Draw the filled portion of the enemy bar
        filled_enemy_bar_rect = pygame.Rect(
            enemy_progress_bar_start_x,
            enemy_bar_y,
            filled_enemy_bar_width,
            enemy_bar_height
        )
        pygame.draw.rect(self.screen, WHITE_COLOR, filled_enemy_bar_rect)  # White fill for current progress

        # Draw vertical dividers for each integer count of enemy eliminations
        for i in range(1, max_num_enemy_eliminations + 1):  # Include the last segment
            segment_x = enemy_progress_bar_start_x + (i / max_num_enemy_eliminations) * enemy_bar_width
            line_color = WHITE_COLOR if i > num_enemy_eliminations else BLACK_COLOR
            pygame.draw.line(
                self.screen,
                line_color,  # Color of the segment divider
                (segment_x, enemy_bar_y),  # Start point of the line
                (segment_x, enemy_bar_y + enemy_bar_height),  # End point of the line
                2  # Line thickness
            )

    
    def human_respawn_handler(self):
        """
        Respawn player agents (Nao, Human, Shutter) after being eliminated
        """
        for agent in Players.values():
            if self.config.game_type == GameTypes.COMPETITIVE_TUTORIAL and Players(agent) not in self.config.interactive_players:
                # Always deactivate non-interactive players in competitive tutorial mode
                self.state.players_state.active[agent] = False
                continue

            if self.state.players_state.active[agent] == False:

                frames_since_elimination = self.state.time_state.frame - self.state.history.last_hit_frame[agent]
                if frames_since_elimination >= DynamicsConsts.PLAYER_RESPAWN_FREQUENCY_FRAMES:
                    self.state.players_state.active[agent] = True
                    self.step_info.players_respawned.append(Players(agent))
                    
                    if agent == Players.HUMAN.value:
                        self.state.players_state.human_position_x = InitialStateConsts.HUMAN_POSITION_X
                        self.state.players_state.human_position_y = InitialStateConsts.HUMAN_POSITION_Y
                    elif agent == Players.SHUTTER.value:
                        self.state.players_state.shutter_position_x = InitialStateConsts.SHUTTER_POSITION_X
                        self.state.players_state.shutter_position_y = InitialStateConsts.SHUTTER_POSITION_Y
                    elif agent == Players.NAO.value:
                        self.state.players_state.nao_position_x = InitialStateConsts.NAO_POSITION_X
                        self.state.players_state.nao_position_y = InitialStateConsts.NAO_POSITION_Y
                    else:
                        raise ValueError("Invalid agent value")

    def render(self, render_mode: RenderModes = None):
        if render_mode is None:
            render_mode = self.config.render_mode
        # bullet_size = .006

        # TODO: fix hacky temp replacement of self.screen variable; current rendering consts depend on original screen dimensions
        old_screen = self.screen
        self.screen = pygame.Surface((RenderConsts.SCREEN_WIDTH, RenderConsts.SCREEN_HEIGHT))

        # Clear the screen
        self.screen.fill((0, 0, 0))  # Black background

        if DRAW_CENTER_AND_SPAWN_LINES:
            # Draw a vertical line down the middle of the screen
            middle_x = RenderConsts.SCREEN_WIDTH // 2
            pygame.draw.line(
                self.screen,  # Surface to draw on
                (255, 255, 255),  # White color
                (middle_x, 0),  # Start point (top of the screen)
                (middle_x, RenderConsts.SCREEN_HEIGHT),  # End point (bottom of the screen)
                2  # Line width (2 pixels)
            )

            # Draw two more lines 0.1*screen width to the right and left of middle lines
            left_x = middle_x - int(DynamicsConsts.SAFE_CENTER_SPAWN_BUFFER_PER_SIDE * RenderConsts.SCREEN_WIDTH)
            right_x = middle_x + int(DynamicsConsts.SAFE_CENTER_SPAWN_BUFFER_PER_SIDE * RenderConsts.SCREEN_WIDTH)
            pygame.draw.line(
                self.screen,  # Surface to draw on
                (255, 255, 255),  # White color
                (left_x, 0),  # Start point (top of the screen)
                (left_x, RenderConsts.SCREEN_HEIGHT),  # End point (bottom of the screen)
                2  # Line width (2 pixels)
            )
            pygame.draw.line(
                self.screen,  # Surface to draw on
                (255, 255, 255),  # White color
                (right_x, 0),  # Start point (top of the screen)
                (right_x, RenderConsts.SCREEN_HEIGHT),  # End point (bottom of the screen)
                2  # Line width (2 pixels)
            )

        # Draw player ships (partially transparent if inactive)
        human_center = (
            int(self.state.players_state.human_position_x * RenderConsts.SCREEN_WIDTH),
            int(self.state.players_state.human_position_y * RenderConsts.SCREEN_HEIGHT)
        )
        shutter_center = (
            int(self.state.players_state.shutter_position_x * RenderConsts.SCREEN_WIDTH),
            int(self.state.players_state.shutter_position_y * RenderConsts.SCREEN_HEIGHT)
        )
        nao_center = (
            int(self.state.players_state.nao_position_x * RenderConsts.SCREEN_WIDTH),
            int(self.state.players_state.nao_position_y * RenderConsts.SCREEN_HEIGHT)
        )
        if self.state.players_state.active['Human']:
            human_rect = self.rendered_objects.ship_image.get_rect(center=human_center)
            self.screen.blit(self.rendered_objects.ship_image, human_rect)
        else:
            human_rect = self.rendered_objects.ship_inactive_image.get_rect(center=human_center)
            self.screen.blit(self.rendered_objects.ship_inactive_image, human_rect)
        if self.state.players_state.active['Shutter']:
            shutter_rect = self.rendered_objects.shutter_image.get_rect(center=shutter_center)
            self.screen.blit(self.rendered_objects.shutter_image, shutter_rect)
        else:
            shutter_rect = self.rendered_objects.shutter_inactive_image.get_rect(center=shutter_center)
            self.screen.blit(self.rendered_objects.shutter_inactive_image, shutter_rect)
        if self.state.players_state.active['Nao']:
            nao_rect = self.rendered_objects.nao_image.get_rect(center=nao_center)
            self.screen.blit(self.rendered_objects.nao_image, nao_rect)
        else:
            nao_rect = self.rendered_objects.nao_inactive_image.get_rect(center=nao_center)
            self.screen.blit(self.rendered_objects.nao_inactive_image, nao_rect)
        
        SHIP_LABEL_COLOR = (255, 255, 255)  # White color for ship labels
        PINK_COLOR = (255, 192, 203)    # Pink for human score
        TURQUOISE_COLOR = (0, 255, 200) # Turquoise for shutter score
        SHIP_LABEL_VERTICAL_OFFSET = 1 # Distance below the ship image to place the label
        
        # Render label for the human player
        human_label_text = "Left"
        human_label_surface = self.rendered_objects.small_font.render(human_label_text, True, PINK_COLOR)
        human_label_rect = human_label_surface.get_rect(midtop=(human_center[0], human_center[1] + RenderConsts.SHIP_HEIGHT // 2 + SHIP_LABEL_VERTICAL_OFFSET))
        self.screen.blit(human_label_surface, human_label_rect)

        # Render label for the Nao robot
        nao_label_text = "Robot"
        nao_label_surface = self.rendered_objects.small_font.render(nao_label_text, True, SHIP_LABEL_COLOR)
        nao_label_rect = nao_label_surface.get_rect(midtop=(nao_center[0], nao_center[1] + RenderConsts.SHIP_HEIGHT // 2 + SHIP_LABEL_VERTICAL_OFFSET))
        self.screen.blit(nao_label_surface, nao_label_rect)

        # Render label for the shutter player
        shutter_label_text = "Right"
        shutter_label_surface = self.rendered_objects.small_font.render(shutter_label_text, True, TURQUOISE_COLOR)
        shutter_label_rect = shutter_label_surface.get_rect(midtop=(shutter_center[0], shutter_center[1] + RenderConsts.SHIP_HEIGHT // 2 + SHIP_LABEL_VERTICAL_OFFSET))
        self.screen.blit(shutter_label_surface, shutter_label_rect)

        # Draw active enemies
        for (x, y), active in zip(zip(InitialStateConsts.ENEMIES_X, InitialStateConsts.ENEMIES_Y), self.state.enemy_state.enemies_active):
            if active:
                leftmost_enemy_x_shutterside = min(InitialStateConsts.SHUTTER_ENEMIES_X)
                if x >= leftmost_enemy_x_shutterside:#0.581:
                    # Use the second enemy image for enemies on the right half
                    enemy_rect = self.rendered_objects.enemy_image2.get_rect(center=(int(x * RenderConsts.SCREEN_WIDTH), int(y * RenderConsts.SCREEN_HEIGHT)))
                    self.screen.blit(self.rendered_objects.enemy_image2, enemy_rect)
                else:
                    # Use the first enemy image for other enemies
                    enemy_rect = self.rendered_objects.enemy_image.get_rect(center=(int(x * RenderConsts.SCREEN_WIDTH), int(y * RenderConsts.SCREEN_HEIGHT)))
                    self.screen.blit(self.rendered_objects.enemy_image, enemy_rect)

        # DEPRECATED: Old single bullet rendering
        # for x, y, shooter in self.state.bullet_state.player_bullets[self.state.bullet_state.playerb_used]:
        #     bullet_rect = self.rendered_objects.bullet_image.get_rect(center=(int(x * RenderConsts.SCREEN_WIDTH), int(y * RenderConsts.SCREEN_HEIGHT)))
        #     self.screen.blit(self.rendered_objects.bullet_image, bullet_rect)

        # Draw player bullets (handling potential multi-bullets)
        for index in self.state.bullet_state.playerb_used:
            bullet_x, bullet_y, shooter = self.state.bullet_state.player_bullets[index]
            multi_bullet_count = self.state.bullet_state.player_multi_bullet_counts[index]
            if multi_bullet_count > 1:
                # draw multi_bullet_count bullets evenly spaced centered at center. Spacing between bullets is RenderConsts.BULLET_WIDTH
                center_x, center_y = int(bullet_x * RenderConsts.SCREEN_WIDTH), int(bullet_y * RenderConsts.SCREEN_HEIGHT)
                bullet_spacing = RenderConsts.BULLET_WIDTH * 1.5

                # 2 bullets:    (center_x - spacing/2), (center_x + spacing/2) --> pos = center_x - spacing/2 + i*spacing = center_x + spacing*(i - 1/2)
                # 3 bullets:    (center_x - spacing), (center_x), (center_x + spacing) --> pos = center_x - spacing + i*spacing
                # 4 bullets:   (center_x - spacing*3/2), (center_x - spacing/2), (center_x + spacing/2), (center_x + spacing*3/2) --> pos = center_x - spacing*3/2 + i*spacing
                # 5 bullets:   (center_x - spacing*2), (center_x - spacing), (center_x), (center_x + spacing), (center_x + spacing*2)
                # n bullets: pos = center_x - spacing*(n-1)/2 + i*spacing
                    # leftmost_x = center_x - spacing*(n-1)/2; pos = leftmost_x + i*spacing
                bullet_x_positions = []
                leftmost_bullet_x = center_x - bullet_spacing * (multi_bullet_count - 1) / 2
                for i in range(multi_bullet_count):
                    bullet_x_positions.append(leftmost_bullet_x + i * bullet_spacing)
                
                for multi_bullet_x in bullet_x_positions:
                    bullet_rect = self.rendered_objects.bullet_image.get_rect(center=(int(multi_bullet_x), center_y))
                    self.screen.blit(self.rendered_objects.bullet_image, bullet_rect)
                    
            elif multi_bullet_count == 1:
                bullet_rect = self.rendered_objects.bullet_image.get_rect(center=(int(bullet_x * RenderConsts.SCREEN_WIDTH), int(bullet_y * RenderConsts.SCREEN_HEIGHT)))
                self.screen.blit(self.rendered_objects.bullet_image, bullet_rect)
            else:
                raise ValueError(f"Invalid multi bullet count: {multi_bullet_count}")

        # Draw enemy bullets (if they are different, load and use another image)
        for x, y in self.state.bullet_state.enemy_bullets[self.state.bullet_state.enemyb_used]:
            enemy_bullet_rect = self.rendered_objects.enemy_bullet_image.get_rect(center=(int(x * RenderConsts.SCREEN_WIDTH), int(y * RenderConsts.SCREEN_HEIGHT)))
            self.screen.blit(self.rendered_objects.enemy_bullet_image, enemy_bullet_rect)
        
        # Render scores
        if SPLIT_SCORE_BAR:
            self.render_scores_split_bar()
        else:
            self.render_scores()
        # nao_score_for_human = int(self.state.score_state.scores['NaoForHuman'])
        # nao_score_for_shutter = int(self.state.score_state.scores['NaoForShutter'])

        # # Render player 1 score
        # p1_score_text = f"YOUR SCORE: {int(self.state.score_state.scores['Human'])}"
        # p1_score_surface = self.rendered_objects.font.render(p1_score_text, True, RenderConsts.SCORE_TEXT_COLOR)  # Green text
        # p1_score_rect = p1_score_surface.get_rect(center=(RenderConsts.SCREEN_WIDTH / 2, RenderConsts.SCORE_BOX_PLAYER1_VERTICAL_OFFSET))
        # self.screen.blit(p1_score_surface, p1_score_rect)
        
        # # Render player 2 score
        # p2_score_text = f"SHUTTER SCORE: {int(self.state.score_state.scores['Shutter'])}"
        # p2_score_surface = self.rendered_objects.font.render(p2_score_text, True, RenderConsts.SCORE_TEXT_COLOR)  # Green text
        # p2_score_rect = p2_score_surface.get_rect(center=(RenderConsts.SCREEN_WIDTH / 2, RenderConsts.SCORE_BOX_PLAYER2_VERTICAL_OFFSET))  # Adjust Y to not overlap with player 1 score
        # self.screen.blit(p2_score_surface, p2_score_rect)

        # Render time progress bar
        if self.config.game_type == GameTypes.COMPETITIVE:
            self.render_time_progress_bar()
        elif self.config.game_type == GameTypes.COMPETITIVE_TUTORIAL:
            self.render_enemy_progress_bar()
        else:
            raise ValueError(f"Unsupported game type {self.config.game_type} for rendering time progress bar.")

        # Render frame count - for testing 
        if self.render_frame_count:
            frame_screen = f"frame: {int(self.state.time_state.frame)}"
            frame_screen_surface = self.rendered_objects.font.render(frame_screen, True, RenderConsts.SCORE_TEXT_COLOR)  # Green text
            frame_screen_rect = frame_screen_surface.get_rect(topleft=(10, RenderConsts.SCORE_BOX_PLAYER2_VERTICAL_OFFSET))  # Adjust Y to not overlap with player 1 score
            self.screen.blit(frame_screen_surface, frame_screen_rect)
        
        # Time
        remaining_seconds = frames_to_seconds(self.config.game_duration_frames - self.state.time_state.frame)
        if self.render_time:
            timer_minutes = remaining_seconds // 60
            timer_seconds = remaining_seconds % 60        
            time_text = f"TIMER {timer_minutes:02}:{timer_seconds:02}"
            time_surface = self.rendered_objects.font.render(time_text, True, RenderConsts.TIME_TEXT_COLOR)  # White text
            time_rect = time_surface.get_rect(topright=(RenderConsts.SCREEN_WIDTH - 10, 10))  # Adjust Y to not overlap with player 1 score
            self.screen.blit(time_surface, time_rect)

        # #Draw bullet rectangels
        # for index in self.state.bullet_state.enemyb_used:
        #     bullet_x, bullet_y = self.state.bullet_state.enemy_bullets[index]
        #     bullet_rect = pygame.Rect(
        #         (bullet_x * RenderConsts.SCREEN_WIDTH)- 2, 
        #         (bullet_y  * RenderConsts.SCREEN_HEIGHT) -6.5,
        #         5,
        #         15
        #     )
        #     sur = (bullet_x *RenderConsts.SCREEN_WIDTH,bullet_y*RenderConsts.SCREEN_HEIGHT)
        #     pygame.draw.rect(self.screen,'white',bullet_rect)

        # Upscale the surface back to desired final size
        game_surface = self.screen
        scaled_surface = pygame.transform.scale(
            game_surface, (
                RenderConsts.SCREEN_WIDTH*RenderConsts.RENDER_UPSCALE_FACTOR,
                RenderConsts.SCREEN_HEIGHT*RenderConsts.RENDER_UPSCALE_FACTOR
            )
        )
        self.screen = old_screen
        self.screen.fill((0, 0, 0))
        self.screen.blit(scaled_surface, (0, 0))
        

        # Update the display
        # if render_mode == RenderModes.RGB_ARRAY.value:
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        #     )
        if render_mode == RenderModes.DISPLAY_WINDOW.value or render_mode == RenderModes.DISPLAY_FULLSCREEN.value:
            pygame.display.flip()
        return self._get_display_as_array()
    
    def _get_display_as_array(self):
        """
        Get the current display as a numpy array
        """
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def reset_config_nao_policy(self):
        nao_y_position = self.state.players_state.nao_position_y

        self.nao_policy_instance = NaoPolicy(
            nao_y_position, RenderConsts.SCREEN_WIDTH, RenderConsts.SCREEN_HEIGHT, DynamicsConsts.VERTICAL_BUFFER,
            DynamicsConsts.HIT_RANGE, DynamicsConsts.SECOND_HIT_RANGE, DynamicsConsts.SHOOTING_RANGE, DynamicsConsts.NAO_RELATIVE_SPEED, DynamicsConsts.FREQUENCY_BOUND_FRAMES, 
            PolicyConsts.POLICY_MODE, #self.config.support_policy
        )
    
    def new_nao_policy(self, new_nao_policy):
        """
        Update the Nao policy
        """
        # nao_x_position = self.state.players_state.nao_position_x
        nao_y_position = self.state.players_state.nao_position_y
        
        self.nao_policy_instance = NaoPolicy(
            nao_y_position, RenderConsts.SCREEN_WIDTH, RenderConsts.SCREEN_HEIGHT, DynamicsConsts.VERTICAL_BUFFER,
            DynamicsConsts.HIT_RANGE, DynamicsConsts.SECOND_HIT_RANGE, DynamicsConsts.SHOOTING_RANGE, DynamicsConsts.NAO_RELATIVE_SPEED, DynamicsConsts.FREQUENCY_BOUND_FRAMES, 
            PolicyConsts.POLICY_MODE, new_nao_policy
        )
