from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env_utils import SpaceInvadersState

from consts import Players

class IdleSupport:
    def __init__(self):
        self.support_player = Players.HUMAN

    def update_support(self, state: 'SpaceInvadersState'):
        return None 