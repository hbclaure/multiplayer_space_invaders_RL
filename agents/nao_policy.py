# agents/nao_policy.py

import time
from .support_policies import (
    BaseSupportPolicy,
    EqualSupport, OnlyHuman, OnlyShutter, EqualizeScores, EqualizeScoresNeutral, 
    EqualizeSupportHistory, IdleSupport, EqualizeSupportHistoryEven,
    BiasedSupportPolicyLeft, BiasedSupportPolicyRight
)


# from .nao_equal_support_policy import EqualSupport
# from .nao_only_support_human import OnlyHuman
# from .nao_only_support_shutter import OnlyShutter
# from .nao_timed_equal_support import TimedEqualSupport
# from .nao_partial_support_human import partialSupportHuman
# from .nao_score_equalizer import EqualizeScores
# from .nao_support_history_equalizer import EqualizeSupportHistory
# from .nao_idle import IdleSupport
import pygame

from env_utils import Players,SpaceInvadersState
from consts import InitialStateConsts, DynamicsConsts, NaoSupportPolicies
from agents.robot_integration import RobotActionRequest, RobotActionType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env_utils import SpaceInvadersState

# TODO: replace init params with consts, then remove init params
# TODO: replace nao_policy() params with state attributes, then remove as arguments

SUPPORT_IDX_TO_PLAYER = {
    0: Players.SHUTTER,
    1: Players.HUMAN,
}

class NaoPolicy:

    def __init__(self, 
                 initial_nao_position_y, 
                 screen_width, 
                 screen_height, 
                 vertical_buffer, 
                 hit_range, 
                 second_hit_range, 
                 shooting_range, 
                 nao_relative_speed, 
                 frequency_bound_frames, 
                 mode,
                 support_policy 
                 ):
        
        self.initial_nao_position_y = initial_nao_position_y
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.vertical_buffer = vertical_buffer
        self.hit_range = hit_range
        self.second_hit_range = second_hit_range
        self.shooting_range = shooting_range
        self.nao_relative_speed = nao_relative_speed
        self.frequency_bound_frames = frequency_bound_frames
        self.mode = mode # 'powerUp' or 'shooting'

        # Instantiate Support Policy
        if support_policy in [
            NaoSupportPolicies.EQUAL_SUPPORT,
            NaoSupportPolicies.TIMED_EQUAL_SUPPORT,
            NaoSupportPolicies.PARTIAL_SUPPORT_HUMAN,
        ]:
            raise ValueError(f"Support policy {support_policy} is deprecated, since it doesn't respond to dynamic state which can be loaded partway through a trajectory.")
        elif support_policy == NaoSupportPolicies.HUMAN_ONLY:
            self.support_policy = OnlyHuman()
        elif support_policy == NaoSupportPolicies.SHUTTER_ONLY:
            self.support_policy = OnlyShutter()
        elif support_policy == NaoSupportPolicies.EQUALIZE_SCORES:
            self.support_policy = EqualizeScores()
        elif support_policy == NaoSupportPolicies.EQUALIZE_SUPPORT_HISTORY:
            self.support_policy = EqualizeSupportHistory()
        elif support_policy == NaoSupportPolicies.EQUALIZE_SUPPORT_HISTORY_EVEN:
            self.support_policy = EqualizeSupportHistoryEven()
        elif support_policy == NaoSupportPolicies.IDLE:
            self.support_policy = IdleSupport()
        elif support_policy == NaoSupportPolicies.EQUALIZE_SCORES_NEUTRAL:
            self.support_policy = EqualizeScoresNeutral()
        elif support_policy == NaoSupportPolicies.BIASED_SUPPORT_LEFT:
            self.support_policy = BiasedSupportPolicyLeft()
        elif support_policy == NaoSupportPolicies.BIASED_SUPPORT_RIGHT:
            self.support_policy = BiasedSupportPolicyRight()
        # TODO: narrowyly support either player winning? keeping score above threshold rather than within like equalize support
        
        else:
            raise Exception(f"An appropriate Nao policy has not been defined. Choose between {list(NaoSupportPolicies)}.")

    def find_nearest_bullet(self, bullets_to_search, nao_position_x):
        
        nearest_bullet = [0, 0]
        x_diff_prev = 1
        for bullet in bullets_to_search:
            x_diff = abs(bullet[0] - nao_position_x)
            #if bullet[1] < self.initial_nao_position_y and bullet[1] > self.vertical_buffer and x_diff < self.hit_range * 2:
            if bullet[1] > self.vertical_buffer and x_diff < self.hit_range * 2:
                if x_diff < x_diff_prev:
                    nearest_bullet = bullet
                    x_diff_prev = x_diff
        return nearest_bullet

    #TODO: implement a both mode
    def nao_policy(self, state: 'SpaceInvadersState', images=None):
        left = False
        right = False
        shoot = False
        
        nao_x = state.players_state.nao_position_x
        previous_support_player = self.support_policy.support_player
        self.support_policy.update_support(state)
        support_player = self.support_policy.support_player
        agent_powerup = {"Shutter": False, "Human": False}
        offset = .005

        # TODO: call nao speech command instead of printing to console
        requested_robot_action = None
        if previous_support_player != support_player or state.time_state.frame == 0:
            POLICY_BASED_SUPPORT_DESCRIPTIONS = True
            if POLICY_BASED_SUPPORT_DESCRIPTIONS:
                description = self.support_policy.describe_support_change(previous_support_player, support_player, state)
                #print(f"Support changed from {previous_support_player} to {support_player}. {description}")
            else:
                description = self.support_policy.describe_policy_agnostic_support_change(previous_support_player, support_player, state)
            print(f"Support changed from {previous_support_player} to {support_player}. Description: {description}")
            
            description_remapped = BaseSupportPolicy.remap_player_names(description)
            
            requested_robot_action = RobotActionRequest(
                    type=RobotActionType.SPEECH,
                    params={"message": description_remapped}
            )

        if support_player == Players.NAO: 
            if nao_x > .5 + offset:
                left, right = True, False
            elif nao_x < .5 - offset:
                left, right = False, True
            else:
                left = right = shoot = False

            nearest_enemy = state.get_nearest_enemy(player=Players.NAO)
            return left, right, shoot, nearest_enemy, support_player, agent_powerup, requested_robot_action

        if self.mode == 'powerUp':  
            nao_y = state.players_state.nao_position_y
            shutter_x = state.players_state.shutter_position_x
            shutter_y = state.players_state.shutter_position_y
            human_x = state.players_state.human_position_x
            human_y = state.players_state.human_position_y

            # hitboxes for collision detection
            nao_rect = images.nao_image.get_rect(center=(int(nao_x * self.screen_width), int(nao_y * self.screen_height)))
            shutter_rect = images.shutter_image.get_rect(center=(int(shutter_x * self.screen_width), int(shutter_y * self.screen_height)))
            human_rect = images.ship_image.get_rect(center=(int(human_x * self.screen_width), int(human_y * self.screen_height)))

            # check for collisions
            nao_human_collision = pygame.Rect.colliderect(nao_rect, human_rect)
            nao_shutter_collision = pygame.Rect.colliderect(nao_rect, shutter_rect)

            # move towards human player if supporting them
            if support_player == Players.HUMAN:
                side_to_search = 'left'
                if nao_x > human_x + offset:
                    left, right = True, False
                elif nao_x < human_x - offset:
                    left, right = False, True
                else:
                    left = right = shoot = False

                if nao_human_collision:
                    agent_powerup["Human"] = True

            # move towards shutter if supporting them
            elif support_player == Players.SHUTTER:
                side_to_search = 'right'
                if nao_x > shutter_x + offset:
                    left, right = True, False
                elif nao_x < shutter_x - offset:
                    left, right = False, True
                else:
                    left = right = shoot = False

                if nao_shutter_collision:
                    agent_powerup["Shutter"] = True
            
            # move towards center if support no one 
            elif support_player == Players.NAO: 
                if nao_x > .5 + offset:
                    left, right = True, False
                elif nao_x < .5 - offset:
                    left, right = False, True
                else:
                    left = right = shoot = False
            else:
                raise ValueError(f"Unsupported player {support_player}. Must be one of {list(Players)}.")

            nearest_enemy = state.get_nearest_enemy(
                player=Players.NAO, 
                side_to_search=side_to_search, 
                excluded_enemies=[]
            )

            return left, right, shoot, nearest_enemy, support_player, agent_powerup, requested_robot_action

        elif self.mode == 'shooting':
            left = False
            right = False
            shoot = False

            nao_position_x = state.players_state.nao_position_x
            bullets_left_side, bullets_right_side = state.get_enemy_bullets_by_side()

            # Update support using the selected support policy
            # NOTE: support_policy.support is deprecated in favor of support_policy.support_player
            self.support_policy.update_support(state)
            support_player = self.support_policy.support_player

            # player_avg_frequency = self.frequency_bound
            # if average_shot_frequency['Human']:
            #     player_avg_frequency = min(self.frequency_bound, average_shot_frequency['Human'])
            
            if support_player == Players.HUMAN:
                bullets_to_search = bullets_left_side
                side_to_search = 'left'
            elif support_player == Players.SHUTTER:
                bullets_to_search = bullets_right_side
                side_to_search = 'right'
            else:
                raise ValueError(f"Support player {support_player} not recognized. Must be one of {list(Players)}.")

            nearest_bullet = self.find_nearest_bullet(bullets_to_search, nao_position_x)

            supported_nearest_enemy = state.get_nearest_enemy(
                player=support_player, side_to_search=side_to_search, excluded_enemies=[]
            )

            # Exclude supported nearest enemy 
            nearest_enemy = state.get_nearest_enemy(
                player=Players.NAO, side_to_search=side_to_search, excluded_enemies=[supported_nearest_enemy]
            )
            # If no nearest enemy is found, ignore player targeting exclusion and try again to target the last enemy
            if nearest_enemy == [0,0]:
                nearest_enemy = state.get_nearest_enemy(
                    player=Players.NAO, side_to_search=side_to_search, excluded_enemies=[]
                )

            # If no nearest enemy is found, default to the middle position of the supported side
            if nearest_enemy == [0, 0]:
                margin = 0.05
                target_position_x = 0.25 if support_player == Players.HUMAN else 0.75
                if nao_position_x < target_position_x - margin:
                    right = True
                elif nao_position_x > target_position_x + margin:
                    left = True
                return left, right, shoot, nearest_enemy, support_player, agent_powerup, requested_robot_action

            # current_time_ms = time.time() * 1000
            # if current_time_ms - last_shot_time['Nao'] < player_avg_frequency * self.nao_relative_speed:
            #     self.nao_shoots = False

            hit = False
            if nearest_bullet[0] <= nao_position_x + self.hit_range and nearest_bullet[0] >= nao_position_x - self.hit_range:
                hit = True

            approached_enemy = False

            if not state.can_shoot(Players.NAO):
                if hit:
                    if nearest_bullet[0] >= 1 - (75 / self.screen_width):
                        left = True
                    elif nearest_bullet[0] <= 55 / self.screen_width:
                        right = True
                    elif nearest_bullet[0] > nao_position_x:
                        left = True
                    elif nearest_bullet[0] <= nao_position_x:
                        right = True
            else:
                if abs(nearest_enemy[0] - nao_position_x) <= self.shooting_range:
                    shoot = True

                if nearest_enemy[0] < nao_position_x:
                
                    if not (nearest_bullet[0] < nao_position_x and self.initial_nao_position_y - nearest_bullet[1] < (200 / self.screen_height) and nao_position_x - nearest_bullet[0] <= self.second_hit_range):
                        left = True
                        approached_enemy = True
                elif nearest_enemy[0] > nao_position_x:
                    if not (nearest_bullet[0] > nao_position_x and self.initial_nao_position_y - nearest_bullet[1] < (200 / self.screen_height) and nearest_bullet[0] - nao_position_x <= self.second_hit_range):
                        right = True
                        approached_enemy = True

                if not approached_enemy and hit:
                    if abs(nearest_bullet[0] - nao_position_x) > 5 / self.screen_width:
                        shoot = False
                    if nearest_bullet[0] >= 1 - (75 / self.screen_width):
                        left = True
                    elif nearest_bullet[0] <= 55 / self.screen_width:
                        right = True
                    elif nearest_bullet[0] > nao_position_x:
                        left = True
                    elif nearest_bullet[0] <= nao_position_x:
                        right = True
            return left, right, shoot, nearest_enemy, support_player, agent_powerup, requested_robot_action

        else:
            raise ValueError(f"Mode: {self.mode} is not supported.")

        
