# Written by: Austin Narcomey
# Date: 2025-06-09

import pygame
import operator
from typing import List, Callable, Type
from abc import ABC, abstractmethod
import copy

from consts import EnhancedEnum
from agents.policies import SingleAgentAction

class PyGameController(ABC):
    """
    Base class for PyGame controllers, providing a common interface for keyboard and gamepad inputs.
    """

    def __init__(self, mapping_type: EnhancedEnum):
        """
        Initialize the controller with a specific mapping type.

        Args:
            mapping_type (EnhancedEnum): The type of mapping for the controller.
        """
        # Check if the subclass has a `MappingTypes` attribute
        if not hasattr(self.__class__, "MappingTypes"):
            raise ValueError(f"{self.__class__.__name__} must define a MappingTypes enum.")

        # Validate the mapping_type if MappingTypes is defined
        if not isinstance(mapping_type, self.__class__.MappingTypes):
            raise ValueError(
                f"mapping_type must be an instance of {self.__class__.MappingTypes.__name__}, "
                f"but got {type(mapping_type).__name__}."
            )

        self.current_action = SingleAgentAction(left=False, right=False, shoot=False)
        self.mapping_type = mapping_type


    @abstractmethod
    def parse_action(self, events: List[pygame.event.Event]) -> SingleAgentAction:
        """
        Parse the current input state into an action.

        Args:
            events (List[pygame.event.Event]): List of pygame events to process.

        Returns:
            SingleAgentAction: The parsed action containing left, right, and shoot commands.
        """
        pass


class AutomatedController(PyGameController):
    """
    No-op controller to stand in for automated players rather than interactive control
    """

    def __init__(self):
        """
        Initialize the automated controller with a no-op action.
        """
        pass

    def parse_action(self, events: List[pygame.event.Event]) -> SingleAgentAction:
        """
        Return a no-op action for automated players.

        Args:
            events (List[pygame.event.Event]): List of pygame events to process.

        Returns:
            SingleAgentAction: A no-op action indicating no movement or shooting.
        """
        return SingleAgentAction(left=False, right=False, shoot=False)
    
    def rumble(self):
        """
        Do nothing for automated players.
        """
        pass

class PyGameKeyboard(PyGameController):
    class MappingTypes(EnhancedEnum):
        LEFT_PLAYER = "left_player"
        RIGHT_PLAYER = "right_player"

    def __init__(self, mapping_type: 'PyGameKeyboard.MappingTypes'):
        super().__init__(mapping_type)
        self.key_mapping = {}
        if mapping_type == PyGameKeyboard.MappingTypes.LEFT_PLAYER:
            self.key_mapping = {
                pygame.K_a: 'left',
                pygame.K_d: 'right',
                pygame.K_w: 'shoot',
            }
        elif mapping_type == PyGameKeyboard.MappingTypes.RIGHT_PLAYER:
            self.key_mapping = {
                pygame.K_LEFT: 'left',
                pygame.K_RIGHT: 'right',
                pygame.K_UP: 'shoot',
            }

    def parse_action(self, events: List[pygame.event.Event]) -> SingleAgentAction:
        """
        Parse the current keyboard state into an action.

        This method updates the action state based on the current keyboard input. 
        Holding movement keys results in continuous movement, as the action state 
        is updated incrementally rather than resetting.

        Args:
            events (List[pygame.event.Event]): List of pygame events to process.

        Returns:
            SingleAgentAction: The parsed action containing left, right, and shoot commands.
        """
        action_dict = self.current_action.to_dict()
        for event in events:
            
            # Ignore non-keyboard events
            if event.type not in [pygame.KEYDOWN, pygame.KEYUP]:
                continue

            # If key pressed, indicate the corresponding action component is true
            # If key released, indicate the corresponding action component is false
            action_component = self.key_mapping.get(event.key, None)
            if action_component is not None:
                # Check if the action component is valid
                if action_component not in action_dict:
                    raise ValueError(f"Invalid action component: {action_component}")
                
                # Update the action dictionary based on the event type (component=left/right/shoot; value=True/False)
                if event.type == pygame.KEYDOWN:
                    action_dict[action_component] = True
                elif event.type == pygame.KEYUP:
                    action_dict[action_component] = False
                
        # Update the current action attributes based on the action_dict
        for action_component, value in action_dict.items():
            setattr(self.current_action, action_component, value)

        return copy.copy(self.current_action)
    

class PyGameGamepad(PyGameController):
    class MappingTypes(EnhancedEnum):
        SPLIT_GAMEPAD_LEFT_PLAYER = "split_gamepad_left_player"
        SPLIT_GAMEPAD_RIGHT_PLAYER = "split_gamepad_right_player"
        SINGLE_GAMEPAD_PER_PLAYER = "single_gamepad_per_player"

    class XboxButtons(EnhancedEnum):
        A = 0
        # TODO: Add other Xbox buttons as needed
        
    
    @classmethod
    def num_gamepads_available(cls) -> int:
        """
        Get the number of gamepads currently available.

        Returns:
            int: The number of gamepads available.
        """
        pygame.joystick.init()
        return pygame.joystick.get_count()

    @classmethod
    def initialize_all_gamepads(cls) -> List['PyGameGamepad']:
        """
        Initialize all available gamepads and return a list of PyGameGamepad instances.

        Returns:
            List[PyGameGamepad]: A list of initialized gamepad instances.
        """
        pygame.joystick.init()
        num_gamepads = pygame.joystick.get_count()
        gamepads = []
        for i in range(num_gamepads):
            gamepad = pygame.joystick.Joystick(i)
            gamepad.init()
            gamepads.append(gamepad)
        return gamepads
    
    def __init__(self, mapping_type: MappingTypes, gamepad_index: int):
        """
        Initialize the gamepad and its mapping based on the mapping type.

        Args:
            index (int): The index of the gamepad to initialize.
            mapping_type (MappingTypes): The type of gamepad mapping.
            verbose (bool): Whether to print initialization details.
        """
        super().__init__(mapping_type)
        
        pygame.joystick.init()
        self.gamepad_index = gamepad_index
        self.gamepad = pygame.joystick.Joystick(gamepad_index)
        self.gamepad.init()
        print(f"Initialized gamepad {gamepad_index}: {self.gamepad.get_name()}")

        # Common thresholds and axes
        self.THRESHOLD_MOVE_LEFT = -0.2
        self.THRESHOLD_MOVE_RIGHT = 0.2
        self.X_AXIS_IN_DPAD = 0  # D-pad left/right is considered as X-axis
        self.Y_AXIS_IN_DPAD = 1  # D-pad up/down is considered as Y-axis

        # Configure mappings based on the mapping type
        if mapping_type == self.MappingTypes.SPLIT_GAMEPAD_LEFT_PLAYER:
            self.X_AXES = [0]  # Left joystick X-axis
            self.DPAD = [0]  # Left D-pad
            self.SHOOT_BUTTONS = [4] + [2, 3]  # Left trigger + top and left action buttons

        elif mapping_type == self.MappingTypes.SPLIT_GAMEPAD_RIGHT_PLAYER:
            self.X_AXES = [3]  # Right joystick X-axis
            self.DPAD = []  # No D-pad for the right player
            self.SHOOT_BUTTONS = [5] + [0, 1]  # Right trigger + bottom and right action buttons

        elif mapping_type == self.MappingTypes.SINGLE_GAMEPAD_PER_PLAYER:
            self.X_AXES = [0, 3]  # Both left and right joystick X-axes
            self.DPAD = [0]  # Left D-pad
            self.SHOOT_BUTTONS = [4, 5] + [0, 1, 2, 3]  # Left/right triggers + all action buttons

        else:
            raise ValueError(f"Invalid mapping type: {mapping_type}")

    def _get_shoot_commands(self) -> bool:
        """
        Get shoot commands from the gamepad.
        Returns:
            bool: True if shoot command is pressed, False otherwise.
        """
        return any(self.gamepad.get_button(btn) == 1 for btn in self.SHOOT_BUTTONS)
    
    def _get_move_commands(self, axes: List[int], dpads: List[int], axis_threshold: float, op: Callable) -> bool:
        """
        Get move commands from the gamepad based on a dynamic operator.

        Args:
            axes (List[int]): List of axes to check.
            dpads (List[int]): List of D-pads to check.
            axis_threshold (float): Threshold value for the axis comparison.
            op (Callable): Comparison operator (e.g., operator.lt, operator.gt).

        Returns:
            bool: True if the move command is pressed, False otherwise.
        """
        if op == operator.lt:
            dpad_value = -1  # Left direction
        elif op == operator.gt:
            dpad_value = 1  # Right direction
        else:
            raise ValueError(f"Unsupported operator: {op}")
        
        axis_condition = [op(self.gamepad.get_axis(axis), axis_threshold) for axis in axes]
        dpad_condition = [
            self.gamepad.get_hat(dpad)[self.X_AXIS_IN_DPAD] == dpad_value for dpad in dpads
        ]
        return any(axis_condition + dpad_condition)

    def _get_move_left_commands(self) -> bool:
        """
        Get move left commands from the gamepad.
        Returns:
            bool: True if move left command is pressed, False otherwise.
        """
        return self._get_move_commands(self.X_AXES, self.DPAD, self.THRESHOLD_MOVE_LEFT, operator.lt)

    def _get_move_right_commands(self) -> bool:
        """
        Get move right commands from the gamepad.
        Returns:
            bool: True if move right command is pressed, False otherwise.
        """
        return self._get_move_commands(self.X_AXES, self.DPAD, self.THRESHOLD_MOVE_RIGHT, operator.gt)

    def parse_action(self, events: List[pygame.event.Event]) -> SingleAgentAction:
        """
        Parse the current gamepad state into an action.

        Returns:
            SingleAgentAction: The parsed action containing left, right, and shoot commands.
        """
        move_left = self._get_move_left_commands()
        move_right = self._get_move_right_commands()
        shoot = self._get_shoot_commands()

        action = SingleAgentAction(
            left=move_left,
            right=move_right,
            shoot=shoot
        )

        return action
    
    def rumble(self):
        """
        Rumble the gamepad if supported.
        This is a placeholder method; actual implementation may vary based on gamepad capabilities.
        """
        try:
            self.gamepad.rumble(
                low_frequency=0.5,
                high_frequency=0.5,
                duration=1000  # Duration in milliseconds
            )
            #print("Gamepad rumbled.")
        except NotImplementedError:
            print("Rumble not supported by this gamepad.")