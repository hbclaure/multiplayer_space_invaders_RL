from typing import TYPE_CHECKING, List, Dict, Type

from interactive_gameplay.utils import (
    GameScriptEvent, GameScript, summarize_scores
)
from consts import Players, EnhancedEnum, DynamicsConsts, ScoreConsts
from agents.robot_integration import RobotPositions
from agents.support_policies import BaseSupportPolicy


if TYPE_CHECKING:
    from env_utils import SpaceInvadersState, SpaceInvadersConfig

# Registry to store available GameScript subclasses
GAME_SCRIPT_REGISTRY: Dict[str, Type[GameScript]] = {}
def register_game_script(name: str):
    """
    Decorator to register a GameScript subclass with a given name.
    """
    def decorator(cls: Type[GameScript]):
        if name in GAME_SCRIPT_REGISTRY:
            raise ValueError(f"Game script '{name}' is already registered.")
        GAME_SCRIPT_REGISTRY[name] = cls
        return cls
    return decorator

@register_game_script("tutorial_left")
def tutorial_left_factory() -> GameScript:
    return TutorialGameScript(player_description="the player on the left")

@register_game_script("tutorial_right")
def tutorial_right_factory() -> GameScript:
    return TutorialGameScript(player_description="the player on the right")

@register_game_script("single_player_left")
def single_player_left_factory() -> GameScript:
    return SinglePlayerGameScript(player_description="the player on the left")

@register_game_script("single_player_right")
def single_player_right_factory() -> GameScript:
    return SinglePlayerGameScript(player_description="the player on the right")

@register_game_script("full_multiplayer")
def full_multiplayer_factory(num_game_segments: int) -> GameScript:
    return FullMultiplayerGameScript(num_game_segments=num_game_segments)


class FullMultiplayerGameScript(GameScript):
    
    SUPPORTED_NUM_SEGMENTS = [3]
        
    class GameStageType(EnhancedEnum):
        START_GAME = "start_game"
        START_SEGMENT = "start_segment"
        END_GAME = "end_game"
    
    # TODO? set the config in a set_config() method rather than pass it in each time, which can then instantiate any necessary variables?
    def __init__(self, num_game_segments: int):
        super().__init__(dynamic_gamepad_assignment=True)
        if num_game_segments not in self.SUPPORTED_NUM_SEGMENTS:
            raise ValueError(f"Unsupported number of game segments: {num_game_segments}. Supported values are: {self.SUPPORTED_NUM_SEGMENTS}.")

        self.num_game_segments = num_game_segments
        #self.powerup_names = ["left_player_double_bullet", "right_player_double_bullet", "none"]
        self.powerup_names = None # Powerups are not supported in the current implementation, so set to None.

    def get_stage(self, state: 'SpaceInvadersState', config: 'SpaceInvadersConfig') -> 'FullMultiplayerGameScript.GameStage':
        """
        Determine the current game stage based on the state.
        """
        num_frames_per_segment = config.game_duration_frames // self.num_game_segments
        current_segment = state.time_state.frame // num_frames_per_segment
        #print(f"Current segment: {current_segment}, Frame: {state.time_state.frame}, Frames per segment: {self.num_frames_per_segment}, Game duration frames: {self.game_duration_frames}")
        if state.is_terminal(config):
            return self.GameStage(self.GameStageType.END_GAME)
        elif current_segment == 0:
            return self.GameStage(self.GameStageType.START_GAME)
        elif 1 <= current_segment < self.num_game_segments:
            # NOTE: starting segment 0 is incorporated into START_GAME, so starting at the next segment 1
            return self.GameStage(self.GameStageType.START_SEGMENT, segment_index=current_segment)
        else:
            raise ValueError(f"Unexpected game stage for frame {state.time_state.frame} in segment {current_segment}.")

    def _get_start_segment_events(self, state: 'SpaceInvadersState', current_segment: int) -> List[GameScriptEvent]:
        if current_segment >= self.num_game_segments:
            raise ValueError(f"Segment index {current_segment} exceeds the number of game segments {self.num_game_segments}.")
        if self.powerup_names is None:
            powerup = "none"
        else:
            powerup = self.powerup_names[current_segment]

        if current_segment > 0:

            score_summary = summarize_scores(score_state=state.score_state)

            ending_last_segment_events = [
                GameScriptEvent(
                    type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                    params={"message": "Let's take a break and answer some questions. I will wait here."}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.MOVE_ROBOT,
                    params={"position_name": RobotPositions.RESTING.value}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                    params={
                        "message": f"Game paused after segment {current_segment} of 3.\n\n{score_summary}\n\nPlease answer the survey questions.",
                        "countdown_seconds": 0,
                        "continue_cue": "keyboard_space",
                    }
                ),
            ]
        else:
            ending_last_segment_events = []
        
        if current_segment == 0:
            starting_speech = "Let's start the game!"
        else:
            starting_speech = "Let's continue the game!" 

        if self.powerup_names is None:
            boost_description = ""
        else:
            #boost_description = f"For this segment of the game, the player on the {'left' if powerup == 'left_player_double_bullet' else 'right'} gets a double bullet boost!"
            if powerup in ["left_player_double_bullet", "right_player_double_bullet"]:
                boost_description = f"For this segment of the game, the player on the {'left' if powerup == 'left_player_double_bullet' else 'right'} gets a double bullet boost!"
            elif powerup == "none":
                boost_description = "For this segment of the game, both players only shoot single bullets."

        # TODO? starting speech, then move robot, then speak boost description and display message?
        starting_new_segment_events = [
            GameScriptEvent(
                type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                #params={"message": f"{starting_speech} {boost_description}"}
                params={"message": starting_speech}
            ),
            GameScriptEvent(
                type=GameScriptEvent.EventType.MOVE_ROBOT,
                params={"position_name": RobotPositions.LOOK_AT_SCREEN.value}
            ),
            GameScriptEvent(
                type=GameScriptEvent.EventType.SET_POWERUP,
                params={"powerup_name": powerup}
            ),
            # GameScriptEvent(
            #     type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
            #     params={
            #         "message": boost_description,
            #         "continue_cue": "all_gamepad_a_button",
            #         "forced_wait_time_seconds": 5,
            #         "countdown_seconds": 5
            #     }
            # ),
        ]
        
        # if no powerups: skip dialog on first segment, then show empty dialog at start of remaining segments
        # if powerups: speech and display dialog for the powerup at start of every segment
        if self.powerup_names is None:
            # Add empty dialog box with cue to wait for players to resume
            # Skipping for segment 0 which already has a similar dialog box describing the policy
            if current_segment > 0:
                starting_new_segment_events += [
                    GameScriptEvent(
                        type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                        params={
                            "message": "",
                            "continue_cue": "all_gamepad_a_button",
                            "forced_wait_time_seconds": 0,
                            "countdown_seconds": 5
                        }
                    )
                ]
        else:
            # Describe the relevant powerup
            starting_new_segment_events += [
                GameScriptEvent(
                    type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                    params={"message": boost_description}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                    params={
                        "message": boost_description,
                        "continue_cue": "all_gamepad_a_button",
                        "forced_wait_time_seconds": 5,
                        "countdown_seconds": 5
                    }
                )
            ]

        return ending_last_segment_events + starting_new_segment_events
    
    def get_stage_events(self, stage: 'FullMultiplayerGameScript.GameStage', state: 'SpaceInvadersState' = None, config: 'SpaceInvadersConfig' = None) -> List[GameScriptEvent]:
        if stage.stage_type == self.GameStageType.START_GAME:
            policy_description = BaseSupportPolicy.describe_policy_from_type(config.support_policy)

            events = [
                GameScriptEvent(
                    type=GameScriptEvent.EventType.MOVE_ROBOT,
                    params={"position_name": RobotPositions.LOOK_AT_SCREEN.value}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                    params={"message": "Can the player on the left press the green button on your controller with the letter Aye?"}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                    params={
                        "message": "Player on the left:",#press the 'A' button on your controller to connect it to the game.",
                        "forced_wait_time_seconds": 5,
                        "continue_cue": "gamepad_a_button",
                    }
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.ASSIGN_CONTROLLER,
                    params={
                        "player_side": "left",
                        "reuse_prior_unblocking_event": True,
                    }
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                    params={"message": "Thanks! Now can the player on the right press the green Aye button on your controller?"}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                    params={
                        "message": "Player on the right:",#press the 'A' button on your controller to connect it to the game.",
                        "forced_wait_time_seconds": 5,
                        "continue_cue": "gamepad_a_button",
                    }
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.ASSIGN_CONTROLLER,
                    params={
                        "player_side": "right",
                        "reuse_prior_unblocking_event": True,
                    }
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                    #params={"message": "Great! Now I know which controller belongs to which player."},# Both of you press the Aye button again to begin."}
                    params={"message": "Great!"},
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                    params={"message": policy_description}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                    # If no powerups, then there won't be a dialog box describing the powerup, so start the countdown here
                    params={
                        "message": policy_description,
                        "forced_wait_time_seconds": 10 if self.powerup_names is None else 10,
                        "countdown_seconds": 5 if self.powerup_names is None else 0,
                        "continue_cue": "all_gamepad_a_button"
                    }
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.MOVE_ROBOT,
                    params={"position_name": RobotPositions.LOOK_AT_SCREEN.value}
                ),
            ]
            events += self._get_start_segment_events(state, current_segment=0)
            return events
        elif stage.stage_type == self.GameStageType.START_SEGMENT:
            current_segment = stage.params["segment_index"]

            return self._get_start_segment_events(state, current_segment=current_segment)

        elif stage.stage_type == self.GameStageType.END_GAME:

            score_summary = summarize_scores(score_state=state.score_state)
            
            if config.independent_victory_score_threshold is None:
                winning_player = state.get_leading_score_player(threshold=0)
                if winning_player == Players.HUMAN:
                    winning_player_description = "The player on the left won, congratulations!"
                elif winning_player == Players.SHUTTER:
                    winning_player_description = "The player on the right won, congratulations!"
                elif winning_player == None:
                    winning_player_description = "The game ended in a draw, well played!"
                else:
                    raise ValueError(f"Unexpected winning player: {winning_player}. Expected one of {Players.HUMAN, Players.SHUTTER, None}.")
            else:
                left_victory = state.get_player_independent_victory(Players.HUMAN, config=config)
                right_victory = state.get_player_independent_victory(Players.SHUTTER, config=config)
                victory_threshold = config.independent_victory_score_threshold

                winning_player_description = ""
                if left_victory:
                    winning_player_description += f"The player on the left scored over {victory_threshold} to win, congratulations!"
                else:
                    winning_player_description += f"The player on the left did not score over {victory_threshold} to win, better luck next time!"
                
                winning_player_description += "\n"
                if right_victory:
                    winning_player_description += f"The player on the right scored over {victory_threshold} to win, congratulations!"
                else:
                    winning_player_description += f"The player on the right did not score over {victory_threshold} to win, better luck next time!"

            return [
                GameScriptEvent(
                    type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                    params={"message": f"{winning_player_description}"}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.MOVE_ROBOT,
                    params={"position_name": RobotPositions.LOOK_AT_SCREEN.value}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                    params={
                        "message": f"Game over after segment {self.num_game_segments} of {self.num_game_segments}:\n\n{score_summary}\n\n{winning_player_description}",
                        "continue_cue": "keyboard_space",
                        "countdown_seconds": 0,
                        "forced_wait_time_seconds": 5
                    }
                ),
            ]

        else:
            raise ValueError(f"Unexpected game stage: {stage}. Expected one of {list(self.GameStage)}.")


class SinglePlayerGameScript(GameScript):
    class GameStage(EnhancedEnum):
        START_GAME = "start_game"
        MID_GAME = "mid_game"
        END_GAME = "end_game"
    
    def __init__(self: int, player_description: str):
        super().__init__(dynamic_gamepad_assignment=True)
        self.player_description = player_description
        if "left" in player_description.lower():
            self.player_side = "left"
        elif "right" in player_description.lower():
            self.player_side = "right"

    def get_stage(self, state: 'SpaceInvadersState', config: 'SpaceInvadersConfig') -> 'SinglePlayerGameScript.GameStage':
        """
        Determine the current game stage based on the state.
        """
        if state.is_terminal(config):
            return self.GameStage.END_GAME
        elif state.time_state.frame == 0:
            return self.GameStage.START_GAME
        elif 0 < state.time_state.frame < config.game_duration_frames:
            return self.GameStage.MID_GAME
        else:
            raise ValueError(f"Unexpected game stage for frame {state.time_state.frame}. Expected frame to be 0, less than {config.game_duration_frames}, or equal to {config.game_duration_frames}.")

    def get_stage_events(self, stage: 'SinglePlayerGameScript.GameStage', state: 'SpaceInvadersState', config: 'SpaceInvadersConfig') -> List['GameScriptEvent']:
        """
        Return the list of events for a given game stage.
        """
        if stage == self.GameStage.START_GAME:
            policy_description = BaseSupportPolicy.describe_policy_from_type(config.support_policy)
            return [
                GameScriptEvent(
                    type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                    params={"message": f"Press the green button on your controller with the letter Aye to begin."}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.MOVE_ROBOT,
                    params={"position_name": RobotPositions.LOOK_AT_SCREEN.value}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                    params={
                        "message": f"Double check that you're holding the controller labelled '{self.player_side}' and we can start the game!",
                        "continue_cue": "gamepad_a_button"
                    }
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.ASSIGN_CONTROLLER,
                    params={
                        "player_side": self.player_side,
                        "reuse_prior_unblocking_event": True,
                    }
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                    params={"message": policy_description}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                    params={
                        "message": policy_description,
                        "forced_wait_time_seconds": 10,
                        "countdown_seconds": 5,
                        "continue_cue": "gamepad_a_button"
                    }
                ),
            ]
        elif stage == self.GameStage.MID_GAME:
            return []
        elif stage == self.GameStage.END_GAME:

            score_summary = summarize_scores(score_state=state.score_state)

            if config.independent_victory_score_threshold is None:
                winning_player = state.get_leading_score_player(threshold=0)
            
                if winning_player == Players.HUMAN:
                    if self.player_side == "left":
                        player_victory = True
                    elif self.player_side == "right":
                        player_victory = False
                elif winning_player == Players.SHUTTER:
                    if self.player_side == "left":
                        player_victory = False
                    elif self.player_side == "right":
                        player_victory = True
                elif winning_player == None:
                    player_victory = None
                else:
                    raise ValueError(f"Unexpected winning player: {winning_player}. Expected one of {Players.HUMAN, Players.SHUTTER, None}.")
                
                if player_victory is True:
                    outcome_message = "Congratulations, you won!"
                elif player_victory is False:
                    outcome_message = "Your opponent won, better luck next time!"
                elif player_victory is None:
                    outcome_message = "The game ended in a draw, well played!"
            else:
                left_victory = state.get_player_independent_victory(Players.HUMAN, config=config)
                right_victory = state.get_player_independent_victory(Players.SHUTTER, config=config)
                victory_threshold = config.independent_victory_score_threshold

                outcome_message = ""
                if left_victory:
                    outcome_message += f"The player on the left scored over {victory_threshold} to win, congratulations!"
                else:
                    outcome_message += f"The player on the left did not score over {victory_threshold} to win, better luck next time!"
                
                outcome_message += "\n"
                if right_victory:
                    outcome_message += f"The player on the right scored over {victory_threshold} to win, congratulations!"
                else:
                    outcome_message += f"The player on the right did not score over {victory_threshold} to win, better luck next time!"

            return [
                GameScriptEvent(
                    type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                    params={"message": f"{outcome_message}"}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.MOVE_ROBOT,
                    params={"position_name": RobotPositions.FACE_TRACK_PLAYERS.value}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                    params={
                        "message": f"Game over:\n\n{score_summary}\n\n{outcome_message}",
                        "continue_cue": "keyboard_space",
                        "countdown_seconds": 0,
                        "forced_wait_time_seconds": 5
                    }
                ),
            ]
        else:
            raise ValueError(f"Unexpected game stage: {stage}. Expected one of {list(self.GameStage)}.")
        
class TutorialGameScript(GameScript):
    # Need start stage to introduce, midgame to just wait, and end game to show concluding message
    
    def __init__(self, player_description: str):
        super().__init__(dynamic_gamepad_assignment=True)
        self.player_description = player_description
        if "left" in player_description.lower():
            self.player_side = "left"
        elif "right" in player_description.lower():
            self.player_side = "right"

    def get_stage(self, state: 'SpaceInvadersState', config: 'SpaceInvadersConfig') -> 'TutorialGameScript.GameStage':
        """
        Determine the current game stage based on the state.
        """
        return super().get_stage(state, config)
    
    def get_stage_events(self, stage: 'TutorialGameScript.GameStage', state: 'SpaceInvadersState', config: 'SpaceInvadersConfig') -> List['GameScriptEvent']:
        """
        Return the list of events for a given game stage.
        """
        if stage.stage_type == self.GameStageType.START_GAME:
            return [
                GameScriptEvent(
                    type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                    params={
                        "message": f"Press the green button on your controller with the letter Aye to begin.",
                    }
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.MOVE_ROBOT,
                    params={"position_name": RobotPositions.LOOK_AT_SCREEN.value}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                    params={
                        "message": f"Double check that you're holding the controller labelled '{self.player_side}' and we can start the tutorial!",
                        "countdown_seconds": 5,
                        "continue_cue": "gamepad_a_button"
                    }
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.ASSIGN_CONTROLLER,
                    params={
                        "player_side": self.player_side,
                        "reuse_prior_unblocking_event": True,
                    }
                ),
            ]

        elif stage.stage_type == self.GameStageType.MID_GAME:
            return []

        elif stage.stage_type == self.GameStageType.END_GAME:
            return [
                GameScriptEvent(
                    type=GameScriptEvent.EventType.SPEAK_MESSAGE,
                    params={"message": f"Now you are ready to play!"}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.MOVE_ROBOT,
                    params={"position_name": RobotPositions.FACE_TRACK_PLAYERS.value}
                ),
                GameScriptEvent(
                    type=GameScriptEvent.EventType.DISPLAY_BLOCKING_MESSAGE,
                    params={
                        "message": f"You eliminated {DynamicsConsts.TUTORIAL_NUM_ENEMIES_TO_ELIMINATE} enemies and passed the tutorial!",
                        "continue_cue": "gamepad_a_button",#"keyboard_space",
                        "countdown_seconds": 0,
                        "forced_wait_time_seconds": 5
                    }
                ),
            ]
        else:
            raise ValueError(f"Unexpected game stage: {stage}. Expected one of {list(self.GameStageType)}.")