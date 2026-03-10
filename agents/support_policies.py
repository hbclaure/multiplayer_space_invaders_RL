
from consts import Players, PolicyConsts, NaoSupportPolicies
import random

from typing import TYPE_CHECKING, Type, Dict, List
if TYPE_CHECKING:
    from env_utils import SpaceInvadersState

class BaseSupportPolicy():
    '''
    Base support policy class
    '''
    # Base support policy class
    def __init__(self, support_player):
        self.support_player = support_player
    
    def update_support(self, state: 'SpaceInvadersState'):
        raise NotImplementedError

    def describe_policy_agnostic_support_change(self, previous_support_player: Players, support_player: Players, state: 'SpaceInvadersState') -> str:
        return f"I'm going to support {support_player.value}."
    
    def describe_support_change(self, previous_support_player: Players, support_player: Players, state: 'SpaceInvadersState') -> str:
        raise NotImplementedError("This method should be implemented in a subclass.")
    
    @classmethod
    def get_policy_mapping(cls):
        return {
            NaoSupportPolicies.HUMAN_ONLY: OnlyHuman,
            NaoSupportPolicies.SHUTTER_ONLY: OnlyShutter,
            NaoSupportPolicies.EQUALIZE_SCORES: EqualizeScores,
            NaoSupportPolicies.EQUALIZE_SUPPORT_HISTORY: EqualizeSupportHistory,
            NaoSupportPolicies.EQUALIZE_SUPPORT_HISTORY_EVEN: EqualizeSupportHistoryEven,
            NaoSupportPolicies.IDLE: IdleSupport,
            NaoSupportPolicies.EQUALIZE_SCORES_NEUTRAL: EqualizeScoresNeutral,
            NaoSupportPolicies.BIASED_SUPPORT_LEFT: BiasedSupportPolicyLeft,
            NaoSupportPolicies.BIASED_SUPPORT_RIGHT: BiasedSupportPolicyRight,
        }
    
    @classmethod
    def describe_policy(self) -> str:
        """
        Returns a description of the support policy.
        This method should be overridden in subclasses to provide specific descriptions.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")
    
    def instantiate_support_policy(support_policy: NaoSupportPolicies) -> 'BaseSupportPolicy':
        """
        Instantiate a subclass of BaseSupportPolicy based on the provided NaoSupportPolicies enumeration.

        Args:
            support_policy (NaoSupportPolicies): The support policy enumeration item.

        Returns:
            BaseSupportPolicy: An instance of the corresponding support policy subclass.

        Raises:
            ValueError: If the provided support policy is deprecated or unsupported.
        """
        if support_policy in [
            NaoSupportPolicies.EQUAL_SUPPORT,
            NaoSupportPolicies.TIMED_EQUAL_SUPPORT,
            NaoSupportPolicies.PARTIAL_SUPPORT_HUMAN,
        ]:
            raise ValueError(f"Support policy {support_policy} is deprecated, as it doesn't respond to dynamic state.")

        policy_mapping = BaseSupportPolicy.get_policy_mapping()
        policy_class = policy_mapping.get(support_policy)
        if policy_class is None:
            raise ValueError(f"Unsupported support policy: {support_policy}")

        return policy_class()  # Instantiate the policy class

    @staticmethod
    def describe_policy_from_type(support_policy: NaoSupportPolicies) -> str:
        policy_mapping = BaseSupportPolicy.get_policy_mapping()
        policy_class: Type[BaseSupportPolicy] = policy_mapping.get(support_policy)
        if policy_class is None:
            raise ValueError(f"Unsupported support policy: {support_policy}")

        return BaseSupportPolicy.remap_player_names(policy_class.describe_policy())
    
    @staticmethod
    def remap_player_names(description: str) -> str:
        description_remapped = description.replace(Players.HUMAN.value, "The player on the left")
        description_remapped = description_remapped.replace(Players.SHUTTER.value, "The player on the right")
        description_remapped = description_remapped.replace(Players.NAO.value, "Neither player")

        return description_remapped

class EqualSupport(BaseSupportPolicy):
    '''
    Every 1000 frames support someone else
    '''
    def __init__(self):
        super().__init__(Players.HUMAN)

    def update_support(self, state: 'SpaceInvadersState'):
        frame = state.time_state.frame
        if frame % 1000 == 0:
            self.support_player = Players.HUMAN if self.support_player == Players.SHUTTER else Players.SHUTTER

class IdleSupport(BaseSupportPolicy):
    '''
    Support no one
    '''
    def __init__(self):
        super().__init__(Players.NAO)

    def update_support(self, state: 'SpaceInvadersState'):
        self.support_player = Players.NAO
    

# TODO? every (interval ~= to other policy support changes) frames, restate intention to support the fixed support player? Same for idle?
class OnlyHuman(BaseSupportPolicy):
    '''
    Only support the human
    '''
    def __init__(self):
        super().__init__(Players.HUMAN)

    def update_support(self, state: 'SpaceInvadersState'):
        self.support_player = Players.HUMAN

class OnlyShutter(BaseSupportPolicy):
    '''
    Only support shutter
    '''
    def __init__(self):
        super().__init__(Players.SHUTTER)

    def update_support(self, state: 'SpaceInvadersState'):
        self.support_player = Players.SHUTTER

class PartialSupportHuman(BaseSupportPolicy):
    '''
    Exclusively support human and then after 1800 frames, exclusively support shutter 
    '''
    def __init__(self):
        super().__init__(Players.HUMAN)

    def update_support(self, state: 'SpaceInvadersState'):
        frame = state.time_state.frame
        if frame <= 1800:
            self.support_player = Players.HUMAN
        else:
            self.support_player = Players.SHUTTER

class EqualizeScores(BaseSupportPolicy):
    '''
    Adjust Nao's support dynamically based on score differences.
    Nao switches support when the score difference exceeds a defined threshold.
    '''
    def __init__(self):
        super().__init__(Players.HUMAN)

    def update_support(self, state: 'SpaceInvadersState'):
        #Uses scores to keep score difference minimal. 
        #Have Nao change support after hitting a threshold in dfferences of scores

        score_threshold = PolicyConsts.SCORE_DIFFERENCE_THRESHOLD #20 ## Threshold: Switch support when the score difference exceeds 20 points
        
        # Retrieve the scores for the Human and Shutter players
        human_score = state.score_state.scores['Human'] + state.score_state.scores['NaoForHuman']
        shutter_score = state.score_state.scores['Shutter'] + state.score_state.scores['NaoForShutter']

        # Calculate the difference between the Human and Shutter scores
        score_difference_shutter_human = human_score- shutter_score

        # If the score difference exceeds the threshold and the human is leading, support Shutter
        if abs(score_difference_shutter_human) >= score_threshold  and  score_difference_shutter_human > 0:
            #self.support = 0
            self.support_player = Players.SHUTTER

        # If the score difference exceeds the threshold and Shutter is leading, support the Human
        elif abs(score_difference_shutter_human) >= score_threshold  and  score_difference_shutter_human < 0: #If shutter's score is greater than threshold, support human. 
            #self.support = 1
            self.support_player = Players.HUMAN

        else:
            #in cases where this doesn't apply just support human. 
            #self.support = 1

            # If the score difference is within the threshold, support the player who nao was previously supporting
            previous_nao_supporting_player = state.players_state.nao_supporting_player
            if previous_nao_supporting_player == Players.HUMAN:
                #self.support = 1
                self.support_player = Players.HUMAN
            elif previous_nao_supporting_player == Players.SHUTTER:
                #self.support = 0
                self.support_player = Players.SHUTTER
            elif previous_nao_supporting_player == Players.NAO:
                # This should happen only at the first step, but if it does, support the human player
                #self.support = 1
                self.support_player = Players.HUMAN
            else:
                raise ValueError(f"Invalid player {previous_nao_supporting_player} for Nao to support.")  
            
    def describe_support_change(self, previous_support_player: Players, support_player: Players, state: 'SpaceInvadersState') -> str:
        score_threshold = PolicyConsts.SCORE_DIFFERENCE_THRESHOLD #20 ## Threshold: Switch support when the score difference exceeds 20 points
        
        # Retrieve the scores for the Human and Shutter players
        human_score = state.score_state.scores['Human'] + state.score_state.scores['NaoForHuman']
        shutter_score = state.score_state.scores['Shutter'] + state.score_state.scores['NaoForShutter']

        # Calculate the difference between the Human and Shutter scores
        score_difference_shutter_human = human_score- shutter_score

        if abs(score_difference_shutter_human) < score_threshold:
            raise ValueError(f"Near tied scores should not have change in support for this policy.")
        elif score_difference_shutter_human > 0:
            return f"{Players.SHUTTER.value} is falling behind, so I'm going to help them catch up and take the lead."
        elif score_difference_shutter_human < 0:
            return f"{Players.HUMAN.value} is falling behind, so I'm going to help them catch up and take the lead."

class EqualizeScoresNeutral(BaseSupportPolicy):
    """
    Adjust Nao's support dynamically based on score difference
    Nao goes to the middle when the score difference exceeds a defined threshold
    """   
    def __init__(self):
        super().__init__(Players.NAO)
        self.last_support_change_frame = 0
    
    def update_support(self, state):
        score_threshold = PolicyConsts.SCORE_DIFFERENCE_THRESHOLD
        min_support_duration = PolicyConsts.MIN_SUPPORT_DURATION_FRAMES

        frames_since_last_change = state.time_state.frame - self.last_support_change_frame
        #print(f"Seconds since last change: {frames_since_last_change / 60:.2f} seconds")
        if frames_since_last_change < min_support_duration:
            return

        human_score = state.score_state.scores['Human'] + state.score_state.scores['NaoForHuman']
        shutter_score = state.score_state.scores['Shutter'] + state.score_state.scores['NaoForShutter']
        score_difference_shutter_human = human_score- shutter_score


        if abs(score_difference_shutter_human) >= score_threshold and score_difference_shutter_human > 0:
            self.support_player = Players.SHUTTER

        elif abs(score_difference_shutter_human) >= score_threshold and score_difference_shutter_human < 0: 
            self.support_player = Players.HUMAN
        
        else:
            self.support_player = Players.NAO

        previous_support_player = state.players_state.nao_supporting_player
        if self.support_player != previous_support_player:
            self.last_support_change_frame = state.time_state.frame

    def describe_support_change(self, previous_support_player: Players, support_player: Players, state: 'SpaceInvadersState') -> str:
        score_threshold = PolicyConsts.SCORE_DIFFERENCE_THRESHOLD
            
        human_score = state.score_state.scores['Human'] + state.score_state.scores['NaoForHuman']
        shutter_score = state.score_state.scores['Shutter'] + state.score_state.scores['NaoForShutter']
        score_difference_shutter_human = human_score- shutter_score

        if abs(score_difference_shutter_human) < score_threshold:
            return f"Scores are close, so I'm going to let you play."
        elif score_difference_shutter_human > 0:
            return f"{Players.SHUTTER.value} is falling behind, so I'm going to help them catch up."# and then step back and let you play."
        elif score_difference_shutter_human < 0:
            return f"{Players.HUMAN.value} is falling behind, so I'm going to help them catch up."# and then step back and let you play."
        
    @classmethod
    def describe_policy(cls) -> str:
        return "For this game, I will support the player that is falling behind, but if you are close then I will step back and let you play."

class EqualizeSupportHistory(BaseSupportPolicy):
    '''
    Keep scores within a certain threshold based on support history. Nao switches support any time the difference in support history exceeds a certain threshold.    
    '''
    def __init__(self, min_frame_count=100):
        #Including a minimum frame coutn to prevent Nao from oscillating in the beginnign of the game. 
        super().__init__(support_player=Players.HUMAN)
        self.min_frame_count = min_frame_count

    def update_support(self, state: 'SpaceInvadersState'):
        # Check if the game is past the minimum frame count. Early exit during the grace period
        if state.time_state.frame < self.min_frame_count:
            return
        
        support_count_history = state.history.support_frame_count
        
        # Calculate the difference in support history
        support_history_difference = support_count_history[Players.HUMAN] - support_count_history[Players.SHUTTER]
        
        # Set the threshold to 25% of the total frames or total support count
        total_support = support_count_history[Players.HUMAN] + support_count_history[Players.SHUTTER]
        threshold = PolicyConsts.SUPPORT_HISTORY_BALANCE_THRESHOLD * total_support#0.25 * total_support

        # Check if the difference exceeds the threshold
        if abs(support_history_difference) > threshold:
            # If Human has received significantly more support, switch to Shutter
            if support_history_difference > 0:
                #self.support = 0
                self.support_player = Players.SHUTTER
            # If Shutter has received significantly more support, switch to Human
            else:
                #self.support = 1
                self.support_player = Players.HUMAN

    def describe_support_change(self, previous_support_player: Players, support_player: Players, state: 'SpaceInvadersState') -> str:
        return f"To support each of you for an equal amount of time, I'm going to help {support_player.name}."

class EqualizeSupportHistoryEven(BaseSupportPolicy):
    """
    Nao switches support after a fixed interval of frames
    """
    def __init__(self, switch_interval_frames=PolicyConsts.EQUAL_SUPPORT_INTERVAL_FRAMES): #
        super().__init__(support_player=Players.NAO)
        self.switch_interval_frames = switch_interval_frames

    def update_support(self, state: 'SpaceInvadersState'):
        frame = state.time_state.frame
        
        # Switch support every switch_interval_frames frames
        if frame == 0:
            random_initial_support = random.choice([Players.HUMAN, Players.SHUTTER])
            self.support_player = random_initial_support
        elif frame % self.switch_interval_frames == 0:
            previous_nao_supporting_player = state.players_state.nao_supporting_player
            if previous_nao_supporting_player == Players.HUMAN:
                self.support_player = Players.SHUTTER
            elif previous_nao_supporting_player == Players.SHUTTER:
                self.support_player = Players.HUMAN
            else:
                raise ValueError(f"Invalid player {previous_nao_supporting_player} for Nao to support.")
            
    def describe_support_change(self, previous_support_player: Players, support_player: Players, state: 'SpaceInvadersState') -> str:
        return f"To support each of you for an equal amount of time, I'm going to help {support_player.value}."

    @classmethod
    def describe_policy(cls) -> str:
        return f"For this game, I will switch between supporting each of you after an equal fixed amount of time."


# TODO: biased support policies toward left or right
    # robot will spend 75/80/90% time toward the favored side
    # switch sides at fixed interval to maintain the support ratio?
    # description of actions says 
        # "To give extra support to favored_player.value, I'm going to help favored_player.value"
        # "To give some support to unfavored_player.value, I'm going to help unfavored_player.value"
    # description of the policy as a whole says
        # "For this game, I will support {favored_player.value} more than {unfavored_player.value}."

    # Support the disfavored player for a fixed interval EQUAL_SUPPORT_INTERVAL_FRAMES,
        # and then support the favored player for EQUAL_SUPPORT_INTERVAL_FRAMES * x where x is a factor to get the desired ratio

        # maybe allow shuffling of order of these intervals: have a parameter for a particular length of time where the ratio
            # should be maintained exactly (e.g. every minute of gameplay should have exactly the ratio of support)
            # but the support intervals can be shuffled random within that constraint
    
class BiasedSupportPolicy(BaseSupportPolicy):
    """
    Biased support policy where Nao spends more time supporting the favored player.
    The ratio of support time between the favored and unfavored players is configurable.
    """
    def __init__(self, favored_player: Players, 
                 unfavored_support_interval_frames: int = PolicyConsts.INTERVAL_TO_SUPPORT_UNFAVORED_PLAYER_FRAMES,
                 interval_to_maintain_biased_ratio_frames = PolicyConsts.INTERVAL_TO_MAINTAIN_BIASED_RATIO_FRAMES
                 ):
        """
        Initialize the biased support policy.

        Args:
            favored_player (Players): The player to favor (e.g., Players.HUMAN or Players.SHUTTER).
            support_ratio (float): The ratio of time spent supporting the favored player (e.g., 0.75 means 75% of the time).
            interval_frames (int): The base interval for switching support.
        """
        super().__init__(support_player=favored_player)
        # Assign attributes
        self.favored_player = favored_player
        self.unfavored_player = Players.HUMAN if favored_player == Players.SHUTTER else Players.SHUTTER
        self.unfavored_support_interval_frames = unfavored_support_interval_frames
        self.interval_to_maintain_biased_ratio_frames = interval_to_maintain_biased_ratio_frames
        
        # Initialize interval tracking
        self.current_interval_start_frame = 0
        self.current_interval_index = 0
        self.shuffled_intervals = self._generate_shuffled_intervals()

    def _generate_shuffled_intervals(self) -> List[dict]:
        """
        Generate a list of shuffled support intervals that maintain the exact support ratio
        over the `balanced_interval_frames` period.

        Returns:
            List[dict]: A list of dictionaries, each containing the player and duration of support.
        """
        unfavored_duration = int(self.unfavored_support_interval_frames)
        favored_duration = int(self.interval_to_maintain_biased_ratio_frames - unfavored_duration)

        # Split durations into intervals
        unfavored_intervals = [{"player": self.unfavored_player, "duration": self.unfavored_support_interval_frames}] * (unfavored_duration // self.unfavored_support_interval_frames)
        favored_intervals = [{"player": self.favored_player, "duration": self.unfavored_support_interval_frames}] * (favored_duration // self.unfavored_support_interval_frames)

        # Combine and shuffle intervals
        all_intervals = unfavored_intervals + favored_intervals
        random.shuffle(all_intervals)

        return all_intervals

    def update_support(self, state: 'SpaceInvadersState'):
        """
        Update the support player based on the biased support policy.
        """
        frame = state.time_state.frame
        frames_since_interval_start = frame - self.current_interval_start_frame

        # Check if the current interval has ended
        if frames_since_interval_start >= self.shuffled_intervals[self.current_interval_index]["duration"]:
            # Move to the next interval
            self.current_interval_index += 1
            if self.current_interval_index >= len(self.shuffled_intervals):
                # Regenerate intervals after completing the balanced period
                self.shuffled_intervals = self._generate_shuffled_intervals()
                self.current_interval_index = 0

            # Update the support player and reset the interval start frame
            self.support_player = self.shuffled_intervals[self.current_interval_index]["player"]
            self.current_interval_start_frame = frame

    def describe_support_change(self, previous_support_player: Players, support_player: Players, state: 'SpaceInvadersState') -> str:
        """
        Describe the reason for the support change.
        """
        if support_player == self.favored_player:
            return f"To give extra support to {self.favored_player.value}, I'm going to help them now."
        else:
            return f"To give some support to {self.unfavored_player.value}, I'm going to help them now."

    @classmethod
    def describe_policy(cls) -> str:
        raise NotImplementedError("This method should be implemented in a subclass.")

class BiasedSupportPolicyLeft(BiasedSupportPolicy):
    """
    Biased support policy favoring the left player (Human).
    """
    def __init__(self):
        super().__init__(favored_player=Players.HUMAN)

    @classmethod
    def describe_policy(cls) -> str:
        return f"For this game, I will support {Players.HUMAN.value} more than {Players.SHUTTER.value}."
    
class BiasedSupportPolicyRight(BiasedSupportPolicy):
    """
    Biased support policy favoring the right player (Shutter).
    """
    def __init__(self):
        super().__init__(favored_player=Players.SHUTTER)

    @classmethod
    def describe_policy(cls) -> str:
        return f"For this game, I will support {Players.SHUTTER.value} more than {Players.HUMAN.value}."


# class TimedEqualSupport:
#     #Nao Switches support at the halfway point or after  5000 frames
#     def __init__(self):
#         self.support = 1

#     def update_support(self, frame,scores,support_count_history):
#         if frame <= 3600:
#             self.support = 1 
#         else:
#             self.support = 0