# agents/nao_only_support_shutter.py

from consts import Players

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env_utils import SpaceInvadersState

class OnlyShutter:
    #Nao only supports Shutter
    def __init__(self):
        #self.support = 0
        self.support_player = Players.SHUTTER

    def update_support(self, state: 'SpaceInvadersState'):
        #self.support =0
        self.support_player = Players.SHUTTER

