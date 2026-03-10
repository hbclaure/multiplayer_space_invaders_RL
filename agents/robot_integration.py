from consts import EnhancedEnum
import os
import json

class RobotActionType(EnhancedEnum):
    """
    Enum for different types of robot actions.
    """
    SPEECH = "speech"
    MOVEMENT = "movement"

class RobotPositions(EnhancedEnum):
    RESTING = "resting"
    LOOK_AT_SCREEN = "look_at_screen_without_blocking"
    FACE_TRACK_PLAYERS = "face_tracker"

class RobotActionRequest:
    def __init__(self, type: RobotActionType, params: dict):
        self.type = type
        self.params = params

    def __str__(self) -> str:
        return f"RobotActionRequest(type={self.type}, params={self.params})"
    
    def __repr__(self) -> str:
        return f"RobotActionRequest(type={self.type}, params={self.params})"

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "params": self.params
        }
    
    @classmethod
    def from_dict(cls, dct: dict) -> "RobotActionRequest":
        return cls(**dct)
    
class RobotActionQueue:
    def __init__(self, save_dir: str):
        self.queue = []
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

    def add_robot_action(self, action: RobotActionRequest):
        """
        Add an action to the queue.
        
        Args:
            action (RequestedRobotAction): The action to add to the queue.
        """

        action_file = os.path.join(self.save_dir, f"robot_action_{len(self.queue)}.json")
        json.dump(action.to_dict(), open(action_file, "w"))
        
        self.queue.append(action)

        

        print("="*80)
        print(f"Robot action added to queue: {action}")
        print("="*80)

