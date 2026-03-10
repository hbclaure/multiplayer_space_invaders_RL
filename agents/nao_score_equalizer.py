# agents/nao_score_equalizer.py

from consts import PolicyConsts, Players

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env_utils import SpaceInvadersState

class EqualizeScores:
    '''
    Adjust Nao's support dynamically based on score differences.
    Nao switches support when the score difference exceeds a defined threshold.
    '''
    def __init__(self):
        #self.support = 1
        self.support_player = Players.HUMAN

    def update_support(self, state: 'SpaceInvadersState'):
        #Uses scores to keep score difference minimal. 
        #Have Nao change support after hitting a threshold in dfferences of scores

        score_threshold = PolicyConsts.SCORE_DIFFERENCE_THRESHOLD #20 ## Threshold: Switch support when the score difference exceeds 20 points
        
        # Retrieve the scores for the Human and Shutter players
        human_score = state.score_state.scores['Human']
        shutter_score = state.score_state.scores['Shutter']

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


        
