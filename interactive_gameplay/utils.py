import pygame
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List  
import time

from consts import EnhancedEnum, InteractiveModes, Players
from models import Trajectory, StepInfo, ActionSpaces, DynamicsConsts
from agents.control_mapping import PyGameKeyboard, PyGameGamepad, AutomatedController
from agents.robot_integration import RobotActionRequest, RobotActionType, RobotPositions

if TYPE_CHECKING:
    from utils import EnvRenderWrapper
    from env_utils import SpaceInvadersState, SpaceInvadersConfig, ScoreState
    from agents.robot_integration import RobotActionQueue

'''
Overall design

    Interactive Trajectory Executor class
        - takes in environment, game script, robot action queue, other parameters

        - instantiates a game script executor object, which is given the environment and robot action queue

        - rollout function that ticks environment, manages interactive controllers, checks the game script,
          and executes game script events by calling the executor

            - more specifically: game script defines which stage of the game we're in and what should happen in each stage
            - the executor is agnostic to any particular script and just executes the events defined by the script

    Game Script Executor class
        - takes in environment, robot action queue, and other parametrs

        - has a method to execute game script events, which will call the robot action queue and environment

    Game Script class
        - base class defining required methods, not much general functionality needed here

    Game Script subclasses
        - for a particular interaction flow, like 3 stage multiplayer game,
        - defines the game stages and events for each stage

    


'''

def summarize_scores(score_state: 'ScoreState') -> str:
    left_score_player = score_state.scores['Human']
    left_score_robot = score_state.scores['NaoForHuman']
    right_score_player = score_state.scores['Shutter']
    right_score_robot = score_state.scores['NaoForShutter']

    score_summary = f"Left Player: {left_score_player + left_score_robot} (with {left_score_robot} from robot)"
    score_summary += f"\nRight Player: {right_score_player + right_score_robot} (with {right_score_robot} from robot)"

    return score_summary


class GameScript(ABC):
    """
    Base class for game scripts that define the flow of the game.
    Subclasses must implement methods to determine the current stage and events for each stage.
    """

    class GameStageType(EnhancedEnum):
        """
        Base class for game stages. It is an enhanced enum that can be extended to define specific game stages.
        
        Subclasses of GameScript can define specific game stages by extending this class.
        """
        START_GAME = "start_game"
        MID_GAME = "mid_game"
        END_GAME = "end_game"

    class GameStage:
        """
        Represents a game stage with a known type and optional parameters for configuration.
        """
        def __init__(self, stage_type: 'GameScript.GameStageType', **params):
            """
            Initialize a GameStage with a type and optional parameters.

            :param stage_type: The type of the stage (e.g., "start_game", "start_segment").
            :param params: Additional parameters to configure the stage.
            """
            self.stage_type = stage_type
            self.params = params

        def __eq__(self, other):
            if isinstance(other, GameScript.GameStage):
                return self.stage_type == other.stage_type and self.params == other.params
            return False

        def __str__(self):
            return f"GameStage(type={self.stage_type}, params={self.params})"

        def __repr__(self):
            return self.__str__()
    
    def __init__(self, dynamic_gamepad_assignment: bool = False):
        """
        Initializes the GameScript base class.
        Subclasses should implement specific game stages and events.
        """
        self.dynamic_gamepad_assignment = dynamic_gamepad_assignment  # Whether to allow dynamic gamepad assignment during the game script execution
    
    def get_stage(self, state: 'SpaceInvadersState', config: 'SpaceInvadersConfig') -> 'GameStage':
        """
        Determine the current game stage based on the state.
        Subclasses can borrow this default implementation or override it.
        """
        if state.is_terminal(config):
            return self.GameStage(self.GameStageType.END_GAME)
        elif state.time_state.frame == 0:
            return self.GameStage(self.GameStageType.START_GAME)
        elif 0 < state.time_state.frame < config.game_duration_frames:
            return self.GameStage(self.GameStageType.MID_GAME)
        else:
            raise ValueError(f"Unrecognized frame state: {state.time_state.frame} for game duration {config.game_duration_frames}.")
        
    def get_pregame_setup_events(self, config: 'SpaceInvadersConfig') -> List['GameScriptEvent']:
        """
        Return the list of events that should be executed before the game starts.
        Subclasses can override this method to provide specific pre-game setup events.
        """
        return []
    
    @abstractmethod
    def get_stage_events(self, stage: 'GameStage', state: 'SpaceInvadersState', config: 'SpaceInvadersConfig') -> List['GameScriptEvent']:
        """
        Return the list of events for a given game stage.
        Subclasses must implement this method.
        """
        pass

class GameScriptEvent:
    """
    Represents an event in the game script that can trigger actions in the environment or robot.
    """
    class EventType(EnhancedEnum):
        DISPLAY_BLOCKING_MESSAGE = "display_message"
        SPEAK_MESSAGE = "speak_message"
        MOVE_ROBOT = "move_robot"
        SET_POWERUP = "set_powerup"
        ASSIGN_CONTROLLER = "assign_controller"

    def __init__(self, type: 'GameScriptEvent.EventType', params: dict):
        """
        Initializes a GameScriptEvent with a specific type and parameters.

        :param type: The type of the event, defined in GameScriptEvent.EventType.
        :param params: A dictionary of parameters specific to the event type.
        """
        self.type = type
        self.params = params

        self.validate_params()

    def validate_params(self):
        """
        Validates the parameters for the event based on its type.
        Raises ValueError if required parameters are missing or invalid.
        """
        # Define required and optional keys for each event type
        allowed_values = {}
        if self.type == GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE:
            required_keys = {"message"}
            optional_keys = {"countdown_seconds", "forced_wait_time_seconds", "continue_cue", "show_continue_cue"}
        
        elif self.type == GameScriptEvent.EventType.SPEAK_MESSAGE:
            required_keys = {"message"}
            optional_keys = {"post_sleep_seconds"}
        
        elif self.type == GameScriptEvent.EventType.MOVE_ROBOT:
            required_keys = {"position_name"}
            optional_keys = {"post_sleep_seconds"}
            allowed_values = {
                "position_name": RobotPositions.values(),
            }
        
        elif self.type == GameScriptEvent.EventType.SET_POWERUP:
            required_keys = {"powerup_name"}
            optional_keys = set()
            allowed_values = {
                "powerup_name": {"left_player_double_bullet", "right_player_double_bullet", "none"}
            }
        
        elif self.type == GameScriptEvent.EventType.ASSIGN_CONTROLLER:
            required_keys = {"player_side"}
            optional_keys = {"reuse_prior_unblocking_event"}
            allowed_values = {
                "player_side": {"left", "right"},
                "reuse_prior_unblocking_event": {True, False},
            }
        
        else:
            raise ValueError(f"Unrecognized event type: {self.type}")

        # Validate the parameters
        self._validate_params(self.params, required_keys, optional_keys, allowed_values)

    @staticmethod
    def _validate_params(params: dict, required_keys: set, optional_keys: set, allowed_values: dict = {}):
        """
        Validates that the provided params contain all required keys and no unrecognized keys.

        :param params: The dictionary of parameters to validate.
        :param required_keys: A set of keys that must be present in params.
        :param optional_keys: A set of keys that are allowed but not required.
        :raises ValueError: If required keys are missing or if unrecognized keys are present.
        """
        allowed_keys = required_keys | optional_keys

        # Check for missing required keys
        missing_keys = required_keys - params.keys()
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        # Check for unrecognized keys
        unrecognized_keys = params.keys() - allowed_keys
        if unrecognized_keys:
            raise ValueError(f"Unrecognized keys: {unrecognized_keys}")

        # Check for allowed values if specified
        for key, allowed in allowed_values.items():
            if key in params and params[key] not in allowed:
                raise ValueError(f"Invalid value for key '{key}': {params[key]}. Allowed values: {allowed}")
    
    def __str__(self):
        return f"GameScriptEvent(type={self.type}, params={self.params})"
    
    def __repr__(self):
        return self.__str__()

class PygameEventWaiter:

    @staticmethod
    def wait_for_all_gamepads_press(button_idx: int = None, timeout_seconds: int = 3600) -> List[pygame.event.Event]:
        num_gamepads = PyGameGamepad.num_gamepads_available()
        pressed_gamepads = set()
        events = []

        while len(pressed_gamepads) < num_gamepads:
            event = PygameEventWaiter.wait_for_gamepad_press(button_idx=button_idx, timeout_seconds=timeout_seconds)
            if event is None:
                raise TimeoutError("Timeout waiting for all gamepads to press the button.")
            pressed_gamepads.add(event.instance_id)
            # print(f"Gamepad {event.instance_id} pressed button {event.button}. Total pressed: {len(pressed_gamepads)}/{num_gamepads}: {pressed_gamepads}")
            events.append(event)
        
        return events
    
    @staticmethod
    def wait_for_gamepad_press(button_idx: int = None, timeout_seconds: int = 3600) -> pygame.event.Event:
        def is_desired_event(event: pygame.event.Event) -> bool:
            # print(f"Processing event: {event}")
            if event.type == pygame.JOYBUTTONDOWN:
                # print(f"Button down event: {event.button} on gamepad {event.joy} (instance_id: {event.instance_id}, looking for button {button_idx})")
                if button_idx is not None:
                    return event.button == button_idx
                return True
            return False

        return PygameEventWaiter._wait_for_pygame_event(
            is_desired_event_callback=is_desired_event, timeout_seconds=timeout_seconds
        )

    @staticmethod
    def wait_for_keyboard_press(key: str = None, timeout_seconds: int = 3600) -> pygame.event.Event:
        if key == "space":
            key = pygame.K_SPACE
        elif key == None:
            pass
        else:
            raise NotImplementedError(f"Keyboard press for key {key} is not implemented. Only 'space' is supported.")
        
        def is_desired_event(event: pygame.event.Event) -> bool:
            if event.type == pygame.KEYDOWN:
                if key is not None:
                    return event.key == key
                return True
            return False

        return PygameEventWaiter._wait_for_pygame_event(
            is_desired_event_callback=is_desired_event, timeout_seconds=timeout_seconds
        )

    @staticmethod
    def _wait_for_pygame_event(is_desired_event_callback, timeout_seconds=3600):
        """
        Wait for a specific pygame event that matches the is_desired_event_callback.
        Returns the first matching event or None if the timeout is reached.
        """
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            events = pygame.event.get()
            for event in events:
                # print(f"Processing event: {event}")
                if is_desired_event_callback(event):
                    return event
        return None

class InteractiveGameExecutor:
    def __init__(self, env: 'EnvRenderWrapper', game_script: 'GameScript' = None, automated_left_player: bool = False,
                 automated_right_player: bool = False, start_state: 'SpaceInvadersState' = None,
                 robot_action_queue: 'RobotActionQueue' = None, debugging_skip_pauses: bool = False, max_len: int = None,
                 trajectory_save_file: str = None):
        self.env = env
        self.game_script = game_script
        self.start_state = start_state
        self.robot_action_queue = robot_action_queue
        self.debugging_skip_pauses = debugging_skip_pauses
        self.max_len = max_len
        self.trajectory_save_file = trajectory_save_file

        self.env_config: 'SpaceInvadersConfig' = env.unwrapped.config

        self.interactive_mode: InteractiveModes = env.unwrapped.config.interactive_mode
        assert self.interactive_mode != InteractiveModes.NONE, "Interactive mode must be specified"

        self.automated_left_player = automated_left_player
        self.automated_right_player = automated_right_player
        interactive_players = env.unwrapped.config.interactive_players
        if self.automated_left_player:
            assert Players.HUMAN not in interactive_players, "Cannot have both automated and interactively controlled left player"
        else:
            assert Players.HUMAN in interactive_players, "Left player must be interactively controlled if not automated"
        if self.automated_right_player:
            assert Players.SHUTTER not in interactive_players, "Cannot have both automated and interactively controlled right player"
        else:
            assert Players.SHUTTER in interactive_players, "Right player must be interactively controlled if not automated"


        action_space = env.unwrapped.config.action_space
        if action_space != ActionSpaces.HUMAN_ONLY:
            raise NotImplementedError("Action space must be HUMAN_ONLY for interactive mode.")

        self.DEBUGGING_ALLOW_ONE_CONTROLLER = True
        self.DEBUGGING_ALLOW_KEYBOARD = True
    
    def _initialize_all_gamepads(self):
        num_gamepads = PyGameGamepad.num_gamepads_available()
        for i in range(num_gamepads):
            PyGameGamepad(gamepad_index=i, mapping_type=PyGameGamepad.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER)
    
    def _default_initialize_controller(self, side: str):
        if side == "left" and self.automated_left_player:
            return AutomatedController()
        elif side == "right" and self.automated_right_player:
            return AutomatedController()
    
        if self.interactive_mode == InteractiveModes.KEYBOARD:
            if side == "left":
                return PyGameKeyboard(mapping_type=PyGameKeyboard.MappingTypes.LEFT_PLAYER)
            elif side == "right":
                return PyGameKeyboard(mapping_type=PyGameKeyboard.MappingTypes.RIGHT_PLAYER)
        elif self.interactive_mode == InteractiveModes.GAMEPAD:
            num_gamepads = PyGameGamepad.num_gamepads_available()
            if num_gamepads == 1 and self.DEBUGGING_ALLOW_ONE_CONTROLLER:
                if side == "left":
                    return PyGameGamepad(gamepad_index=0, mapping_type=PyGameGamepad.MappingTypes.SPLIT_GAMEPAD_LEFT_PLAYER)
                elif side == "right":
                    return PyGameGamepad(gamepad_index=0, mapping_type=PyGameGamepad.MappingTypes.SPLIT_GAMEPAD_RIGHT_PLAYER)
            elif num_gamepads >= 2:
                # Currently on the cart, the left usb port corresponds to index 1, and the right to index 0
                if side == "left":
                    return PyGameGamepad(gamepad_index=1, mapping_type=PyGameGamepad.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER)
                elif side == "right":
                    return PyGameGamepad(gamepad_index=0, mapping_type=PyGameGamepad.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER)
            else:
                if self.DEBUGGING_ALLOW_ONE_CONTROLLER:
                    raise RuntimeError("No gamepads available, but debugging allows one controller.")
                else:
                    raise RuntimeError("No gamepads available. Please connect at least one gamepad.")
        else:
            raise NotImplementedError(f"Interactive mode must be KEYBOARD or GAMEPAD, not {self.interactive_mode}")

    def _dynamically_reassign_controller(self, event: 'GameScriptEvent', unblock_pygame_event: pygame.event.Event = None, check_duplicate_assignments: bool = True):
        """
        Record which controller pressed a specific button.
        """
        if event.type != GameScriptEvent.EventType.ASSIGN_CONTROLLER:
            raise ValueError(f"Event type must be ASSIGN_CONTROLLER, not {event.type}")
        
        if not self.game_script.dynamic_gamepad_assignment:
            raise RuntimeError("Dynamic gamepad assignment is not enabled. Please set DYNAMIC_GAMEPAD_ASSIGNMENT to True.")

        player_side = event.params["player_side"]
        if player_side == "left" and self.automated_left_player:
            raise RuntimeError("Cannot dynamically assign a controller to the left player when they are automated.")
        elif player_side == "right" and self.automated_right_player:
            raise RuntimeError("Cannot dynamically assign a controller to the right player when they are automated.")   
        elif player_side not in ["left", "right"]:
            raise ValueError(f"Unrecognized player side: {event.params['player_side']}. Must be 'left' or 'right'.")

        if event.params.get("reuse_prior_unblocking_event", False):
            if unblock_pygame_event is None:
                raise RuntimeError("No prior unblocking event provided, but use_prior_unblocking_event is True.")
            # Push the event back onto the pygame event queue to pretend it just happened
            pygame.event.post(unblock_pygame_event)

        if self.DEBUGGING_ALLOW_KEYBOARD and self.interactive_mode == InteractiveModes.KEYBOARD:
            PygameEventWaiter.wait_for_keyboard_press(
                key="space", timeout_seconds=10
            )
        elif self.interactive_mode == InteractiveModes.GAMEPAD:
            while True:
                buttonpress_event = PygameEventWaiter.wait_for_gamepad_press(
                    button_idx=PyGameGamepad.XboxButtons.A.value
                )
                gamepad_id = buttonpress_event.instance_id  # Get the gamepad index from the event

                new_controller = PyGameGamepad(gamepad_index=gamepad_id, mapping_type=PyGameGamepad.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER)

                if player_side == "left":
                    if check_duplicate_assignments and not self.automated_right_player and self.right_controller is not None:
                        # Right controller is already assigned, so we cannot assign the same gamepad to the left player
                        if new_controller.gamepad_index == self.right_controller.gamepad_index:
                            # Retry waiting for a different gamepad to push a button; require a new gamepad event 
                            continue
                    self.left_controller = new_controller
                    break
                elif player_side == "right":
                    if check_duplicate_assignments and not self.automated_left_player and self.left_controller is not None:
                        # Left controller is already assigned, so we cannot assign the same gamepad to the right player
                        if new_controller.gamepad_index == self.left_controller.gamepad_index:
                            # Retry waiting for a different gamepad to push a button; require a new gamepad event
                            continue
                    self.right_controller = new_controller
                    break
        else:
            raise NotImplementedError(f"Interactive mode must be KEYBOARD (for development only) or GAMEPAD, not {self.interactive_mode}")

    def _validate_controller_assignment(self):
        if self.left_controller is None:
            raise RuntimeError("Left controller is not initialized. Please assign a gamepad or use keyboard controls.")
        if self.right_controller is None:
            raise RuntimeError("Right controller is not initialized. Please assign a gamepad or use keyboard controls.")
        
        if self.automated_left_player:
            if not isinstance(self.left_controller, AutomatedController):
                raise RuntimeError("Left controller must be an AutomatedController when the left player is automated.")
        else:
            if not isinstance(self.left_controller, PyGameKeyboard) and not isinstance(self.left_controller, PyGameGamepad):
                raise RuntimeError("Left controller must be a PyGameKeyboard or PyGameGamepad when the left player is not automated.") 
        if self.automated_right_player:
            if not isinstance(self.right_controller, AutomatedController):
                raise RuntimeError("Right controller must be an AutomatedController when the right player is automated.")
        else:
            if not isinstance(self.right_controller, PyGameKeyboard) and not isinstance(self.right_controller, PyGameGamepad):
                raise RuntimeError("Right controller must be a PyGameKeyboard or PyGameGamepad when the right player is not automated.")
        
        if self.interactive_mode == InteractiveModes.GAMEPAD:
            if not self.DEBUGGING_ALLOW_ONE_CONTROLLER:
                if self.left_controller.gamepad.get_instance_id() == self.right_controller.gamepad.get_instance_id():
                    raise RuntimeError("Both players are using the same gamepad. Please connect two separate gamepads.")
    
    def _execute_game_script_events(self, game_script_events: List['GameScriptEvent']):
        unblock_pygame_events = []
        for event in game_script_events:
            if event.type == GameScriptEvent.EventType.ASSIGN_CONTROLLER:
                if len(unblock_pygame_events) == 1:
                    # only one event from one gamepad, so we can use it uniquely identify
                    # waiting on all gamepads will return multiple events
                    unblock_pygame_event = unblock_pygame_events[0]
                else:
                    unblock_pygame_event = None

                self._dynamically_reassign_controller(event, unblock_pygame_event)
            elif event.type == GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE:
                if self.debugging_skip_pauses:
                    countdown_seconds = 0
                    forced_wait_time = 0
                else:
                    countdown_seconds = event.params.get("countdown_seconds", 0)
                    forced_wait_time = event.params.get("forced_wait_time_seconds", 0)
                
                unblock_pygame_events = self.env.unwrapped.display_blocking_dialog_message(
                    message=event.params["message"],
                    continue_cue=event.params.get("continue_cue", "keyboard_space"),
                    show_continue_cue=event.params.get("show_continue_cue", True),
                    countdown_seconds=countdown_seconds,
                    forced_wait_time_seconds=forced_wait_time,
                )
            elif event.type == GameScriptEvent.EventType.SPEAK_MESSAGE:
                self.robot_action_queue.add_robot_action(RobotActionRequest(
                    type=RobotActionType.SPEECH,
                    params={"message": event.params["message"]},
                ))
                if event.params.get("post_sleep_seconds", 0) > 0:
                    time.sleep(event.params["post_sleep_seconds"])
            elif event.type == GameScriptEvent.EventType.MOVE_ROBOT:
                self.robot_action_queue.add_robot_action(RobotActionRequest(
                    type=RobotActionType.MOVEMENT,
                    params={"position_name": event.params["position_name"]},
                ))
                if event.params.get("post_sleep_seconds", 0) > 0:
                    time.sleep(event.params["post_sleep_seconds"])
            elif event.type == GameScriptEvent.EventType.SET_POWERUP:
                self.env.unwrapped.set_boost(event.params["powerup_name"],)
            else:
                raise NotImplementedError(f"Unrecognized event type: {event.type}")
        

    def execute(self) -> 'Trajectory':

        # import pdb; pdb.set_trace()
        # pygame.init()
        # gamepads = PyGameGamepad.initialize_all_gamepads()
        # print(gamepads)
        # while True:
        #     events = pygame.event.get()
        #     for event in events:
        #         print(f"Processing event: {event}")


        trajectory = Trajectory(config=self.env_config)
        terminated = False
        previous_game_stage = None

        # Initialize the environment
        obs, info = self.env.reset()
        if self.start_state is not None:
            self.env.set_state(self.start_state)
        state: SpaceInvadersState = self.env.get_state()

        # Initialize controllers (temporarily required before dynamic reassignment)
        # self._initialize_all_gamepads() # Needed if later on dynamically re-assigning
        # PygameEventWaiter.wait_for_all_gamepads_press()

        ### This worked!
        # import pdb; pdb.set_trace()
        # pygame.init() # still work if you remove this
        # gamepads = PyGameGamepad.initialize_all_gamepads()
        # print(gamepads)
        # while True:
        #     events = pygame.event.get()
        #     for event in events:
        #         print(f"Processing event: {event}")
        ###

        ### This worked (alternating which gamepad)
        # import pdb; pdb.set_trace()
        # pygame.init()
        # gamepads = PyGameGamepad.initialize_all_gamepads()
        # print(gamepads)
        # PygameEventWaiter.wait_for_gamepad_press(timeout_seconds=30)
        # PygameEventWaiter.wait_for_gamepad_press(timeout_seconds=30)  # Wait for the first gamepad to press the button
        ###

        ### This worked
        # import pdb; pdb.set_trace()
        # # pygame.init() # this can be removed or added
        # gamepads = PyGameGamepad.initialize_all_gamepads()
        # print(gamepads)
        # PygameEventWaiter.wait_for_all_gamepads_press(timeout_seconds=30)
        ###

        ### either controller init ruined event listening, worked without
        # import pdb; pdb.set_trace()
        # PyGameGamepad.initialize_all_gamepads()  # Initialize all gamepads
        # #self.left_controller = self._initialize_controller("left")
        # #self.right_controller = self._initialize_controller("right")
        # PygameEventWaiter.wait_for_all_gamepads_press(timeout_seconds=30)  # Wait for all gamepads to be pressed
        ###

        ### this was fine
        # import pdb; pdb.set_trace()
        # PyGameGamepad.initialize_all_gamepads()  # Initialize all gamepads
        # #self.left_controller = self._initialize_controller("left")
        # #self.right_controller = self._initialize_controller("right")
        # left = PyGameGamepad(gamepad_index=0, mapping_type=PyGameGamepad.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER)
        # right = PyGameGamepad(gamepad_index=1, mapping_type=PyGameGamepad.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER)
        # PygameEventWaiter.wait_for_all_gamepads_press(timeout_seconds=30)  # Wait for all gamepads to be pressed
        ###

        ### this was also fine
        # import pdb; pdb.set_trace()
        # PyGameGamepad.initialize_all_gamepads()  # Initialize all gamepads
        # self.left_controller = PyGameGamepad(gamepad_index=0, mapping_type=PyGameGamepad.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER)
        # self.right_controller = PyGameGamepad(gamepad_index=1, mapping_type=PyGameGamepad.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER)
        # PygameEventWaiter.wait_for_all_gamepads_press(timeout_seconds=30)  # Wait for all gamepads to be pressed
        ###

        ### commented out get num gamepads in initialize_controller, only one controller worked
        # import pdb; pdb.set_trace()
        # PyGameGamepad.initialize_all_gamepads()  # Initialize all gamepads
        # self.left_controller = self._initialize_controller("left")
        # self.right_controller = self._initialize_controller("right")
        # PygameEventWaiter.wait_for_all_gamepads_press(timeout_seconds=30)  # Wait for all gamepads to be pressed
        ###

        ### didn't work -- it needs both gamepads to be initialized!!!
        # import pdb; pdb.set_trace()
        # PyGameGamepad.initialize_all_gamepads()  # Initialize all gamepads
        # #self.left_controller = PyGameGamepad(gamepad_index=0, mapping_type=PyGameGamepad.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER)
        # #self.right_controller = AutomatedController() # didn't work with or without this
        # PygameEventWaiter.wait_for_all_gamepads_press(timeout_seconds=30)  # Wait for all gamepads to be pressed
        ###

        ### didn't work without gamepads= -- it needs both gamepads to be initialized!!!
        # import pdb; pdb.set_trace()
        # gamepads = PyGameGamepad.initialize_all_gamepads()  # Initialize all gamepads
        # #self.left_controller = PyGameGamepad(gamepad_index=0, mapping_type=PyGameGamepad.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER)
        # #self.right_controller = AutomatedController() # didn't work with or without this
        # PygameEventWaiter.wait_for_all_gamepads_press(timeout_seconds=30)  # Wait for all gamepads to be pressed
        ###




        
        # pygame.init()
        # PyGameGamepad.initialize_all_gamepads()
        # import pdb; pdb.set_trace()
        # PygameEventWaiter.wait_for_all_gamepads_press()
        # print("All gamepads pressed the button. Proceeding with controller initialization.")
        # self.left_controller = self._initialize_controller("left")
        # self.right_controller = self._initialize_controller("right")
        # PygameEventWaiter.wait_for_all_gamepads_press(timeout_seconds=10)  # Wait for all gamepads to be pressed
        
        # if self.DYNAMIC_CONTROLLER_ASSIGNMENT:
        #     self._initialize_all_gamepads()
        
        # if self.DYNAMIC_CONTROLLER_ASSIGNMENT:
        #     self._validate_controller_assignment()

        if self.DEBUGGING_ALLOW_KEYBOARD and self.interactive_mode == InteractiveModes.KEYBOARD:
            self.left_controller = AutomatedController() if self.automated_left_player else PyGameKeyboard(mapping_type=PyGameKeyboard.MappingTypes.LEFT_PLAYER)
            self.right_controller = AutomatedController() if self.automated_right_player else PyGameKeyboard(mapping_type=PyGameKeyboard.MappingTypes.RIGHT_PLAYER)
        elif self.game_script is not None and self.game_script.dynamic_gamepad_assignment:
            self.all_gamepads = PyGameGamepad.initialize_all_gamepads()

            # Controllers will be dynamically assigned in game script events
            self.left_controller = AutomatedController() if self.automated_left_player else None
            self.right_controller = AutomatedController() if self.automated_right_player else None
        else:
            # Initialize controllers for left and right players (which may be gamepads, keyboard, or automated)
            self.left_controller = self._default_initialize_controller("left")
            self.right_controller = self._default_initialize_controller("right")
        
        # Main game loop
        # game_start_events = self.game_script.get_stage_events(GameStage.START_GAME, state)
        # self.game_script_executor.execute_events(game_start_events)
        while not terminated:
            # Enforce frame rate
            self.env.unwrapped.clock.tick(DynamicsConsts.FRAMES_PER_SECOND)

            # Get the current state
            state = self.env.get_state()

            # Determine the current game stage based on current state (if there's a script)
            # NOTE: This will execute on the final state of the trajectory that sets terminated=True and ends the loop
            # NOTE: transition from the last state will render on screen briefly before disappearing
            if self.game_script is not None:
                game_stage = self.game_script.get_stage(state, self.env_config)
                if game_stage != previous_game_stage:
                    # Intermediate save
                    if self.trajectory_save_file is not None:
                        trajectory.to_file(
                            traj_file=self.trajectory_save_file,
                        )

                    # Execute the game script events for the new stage
                    game_stage_events = self.game_script.get_stage_events(game_stage, state, self.env_config)
                    self._execute_game_script_events(game_stage_events)
            else:
                game_stage = None

            # Get actions for each player
            if self.game_script is not None and self.game_script.dynamic_gamepad_assignment:
                self._validate_controller_assignment()  # Ensure controllers are properly assigned before parsing actions
            events = pygame.event.get()
            left_player_action = self.left_controller.parse_action(events)
            right_player_action = self.right_controller.parse_action(events)

            joint_action = {
                'left_player_action_index': left_player_action.tie_breaker(inplace=False).index(),
                'right_player_action_index': right_player_action.tie_breaker(inplace=False).index(),
            }

            # Step the environment
            next_obs, reward, terminated, truncated, info = self.env.step(joint_action)
            trajectory.add_step(state, left_player_action, reward, info)

            # Handle robot actions
            step_info: 'StepInfo' = info["step_info"]
            if self.robot_action_queue is not None and step_info.requested_robot_action is not None:
                self.robot_action_queue.add_robot_action(step_info.requested_robot_action)
            
            # Handle gamepad rumble feedback
            if self.interactive_mode == InteractiveModes.GAMEPAD:
                if Players.HUMAN in step_info.players_hit:
                    self.left_controller.rumble()
                if Players.SHUTTER in step_info.players_hit:
                    self.right_controller.rumble()

            # Optional early termination
            if self.max_len is not None and len(trajectory) >= self.max_len:
                break

            previous_game_stage = game_stage
        
        if self.game_script is None:
            # If no game script is defined, display a blocking message to close the game window
            self._execute_game_script_events([
                GameScriptEvent(
                    type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                    params={"message": "Press the space bar to close the game window", "forced_wait_time_seconds": 5}
                )
            ])

        return trajectory
