# agents/nao_only_support_human.py

from consts import Players

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env_utils import SpaceInvadersState

class OnlyHuman:
    #Nao only supports human 
    def __init__(self):
        #self.support = 1
        self.support_player = Players.HUMAN

    def update_support(self, state: 'SpaceInvadersState'):
        #self.support = 1
        self.support_player = Players.HUMAN

