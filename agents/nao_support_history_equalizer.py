# agents/nao_using_support_history.py
from consts import PolicyConsts
from consts import Players

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env_utils import SpaceInvadersState

class EqualizeSupportHistory:
    '''
    Keep scores within a certain threshold based on support history. Nao switches support any time the difference in support history exceeds a certain threshold.    
    '''
    def __init__(self,min_frame_count=100):
        #Including a minimum frame coutn to prevent Nao from oscillating in the beginnign of the game. 
        #self.support = 1
        self.support_player = Players.HUMAN
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


