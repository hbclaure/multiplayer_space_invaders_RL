import jax.numpy as jnp 
import jax
from .support_policies import *
from env.jx_env_state_params import EnvState, EnvParams
from consts import NaoSupportPolicies
from flax import struct
from typing import Tuple
import functools

class JxBaseActionPolicy:
    '''
    Base action policy class
    '''

    # TODO: Modes
    # ~~TODO: Replace wheres and selects with jax.lax.cond where relevant~~
    # TODO: Verify action selection is consistent with original. Some deaths seem a bit dumb

    def __init__(self):
        pass

    def get_nearest_bullet(self, state: EnvState,  params: EnvParams, side_idx: int, player_idx: int):
        # TODO: Verify this matches the original logic
        # The nearest bullet is any bullet who's y is greater than vertical buffer and x_diff is less than 2 * hit range
        # that is minimal in the x diff 
        bullet_positions = state.bullet_state.enemy_bullet_positions[:, :, side_idx]
        bullet_xs = bullet_positions[:, params.x_idx]
        bullet_ys = bullet_positions[:, params.y_idx]
        player_xs = state.player_state.player_positions[player_idx, params.x_idx]
        player_ys = state.player_state.player_positions[player_idx, params.y_idx]

        x_dists = jnp.abs(bullet_xs - player_xs)
        y_dists = jnp.abs(bullet_ys - player_ys)
        
        y_cond = jnp.greater(bullet_ys, params.dynamics_consts.vertical_buffer)
        x_cond = jnp.less(x_dists, 2 * params.dynamics_consts.hit_range)

        valid_mask = jnp.logical_and(x_cond, y_cond)

        x_dists = jax.lax.select(
            valid_mask, 
            x_dists, 
            jnp.full_like(x_dists, jnp.inf)
        ) 

        y_dists = jax.lax.select(
            valid_mask, 
            y_dists, 
            jnp.full_like(y_dists, jnp.inf)
        ) 

        min_y_dist = jnp.min(y_dists)

        # Mask all y dists not equal to min y dist
        equal_min_y_mask = jnp.equal(y_dists, min_y_dist)

        # Only x dists where y is equal to min y dist
        x_dists_where_y_dist_is_min = jax.lax.select(
            equal_min_y_mask, 
            x_dists, 
            jnp.full_like(a=x_dists, fill_value=jnp.inf, dtype=float))
                
        # Mask all x dists where y isn't equal to min y dist
        x_dists_where_y_dist_is_min_mask = jnp.equal(x_dists, jnp.min(x_dists_where_y_dist_is_min))

        # 1 everywhere y is minimal and x is minimal where y is minimal, 0 otherwise 
        final_mask = jnp.logical_and(equal_min_y_mask, x_dists_where_y_dist_is_min_mask)
        nearest_bullet_idx = jnp.argmax(final_mask)
        
        # nearest_idx = jnp.argmin(x_dists) 
        # Have to flatten because we don't specify axis 
        x = bullet_xs.flatten()[nearest_bullet_idx]
        y = bullet_ys.flatten()[nearest_bullet_idx]
        return jnp.array([x, y])
    
    def get_nearest_enemy(self, state: EnvState, params: EnvParams, side_idx: int, player_idx: int):
        enemy_positions = state.enemy_state.enemy_positions[:, side_idx]  # (2, 5, 5)
        active_enemies = state.enemy_state.enemies_active[side_idx]
        active_enemies_broadcasted = jnp.broadcast_to(active_enemies, shape=enemy_positions.shape) # (2, 5, 5)

        # Mask based on whether enemy is active
        active_only_enemy_positions = jax.lax.select(active_enemies_broadcasted, 
                                                     enemy_positions,
                                                     jnp.full_like(a=enemy_positions, fill_value=jnp.inf, dtype=float))
        
        # Get distances 
        player_pos = state.player_state.player_positions[player_idx] # (2,)
        x_dists = jnp.abs(active_only_enemy_positions[params.x_idx] - player_pos[params.x_idx])
        y_dists = jnp.abs(active_only_enemy_positions[params.y_idx] - player_pos[params.y_idx])
        min_y_dist = jnp.min(y_dists)

        # Mask all y dists not equal to min y dist
        equal_min_y_mask = jnp.equal(y_dists, min_y_dist)

        # Only x dists where y is equal to min y dist
        x_dists_where_y_dist_is_min = jax.lax.select(equal_min_y_mask, 
                                                     x_dists, 
                                                     jnp.full_like(a=x_dists, fill_value=jnp.inf, dtype=float))
                
        # Mask all x dists where y isn't equal to min y dist
        x_dists_where_y_dist_is_min_mask = jnp.equal(x_dists, jnp.min(x_dists_where_y_dist_is_min))

        # 1 everywhere y is minimal and x is minimal where y is minimal, 0 otherwise 
        final_mask = jnp.logical_and(equal_min_y_mask, x_dists_where_y_dist_is_min_mask)
        nearest_enemy_idx = jnp.argmax(final_mask)

        # Construct nearest enemy. Flatten because no axis specified in argmax
        nearest_enemy_x = active_only_enemy_positions[params.x_idx].flatten()[nearest_enemy_idx]
        nearest_enemy_y = active_only_enemy_positions[params.y_idx].flatten()[nearest_enemy_idx]
        nearest_enemy_position = jnp.array([nearest_enemy_x, nearest_enemy_y])
        return nearest_enemy_position

    def can_shoot(self, state: EnvState, params: EnvParams, player_idx: int):
        # TODO: Put these constants in params, also why do we compute it like this?
        # ~~Players can shoot if they're active AND~~ NOTE: This condition has been moved to the action_policy 
        # it's been sufficient time between there shots AND 
        # there are less active bullets then the max bullets  
        # TODO: Am I properly tracking the player_last_shot_frame variable?

        # cond_one = jnp.equal(state.player_state.players_active[player_idx], 1)

        # An agent can shoot if the time since they last shot is greater than a threshold 
        cond_one = jnp.greater(
            state.time_state.frame_num - state.history.player_last_shot_frame[player_idx],
            params.dynamics_consts.min_player_shoot_frequency_frames
        )
        
        num_active_bullets = jnp.count_nonzero(state.bullet_state.player_bullet_actives[player_idx, :])
        cond_two = jnp.less(num_active_bullets, state.bullet_state.max_bullets_allowed[player_idx])

        cond_one_and_two = jnp.logical_and(cond_one, cond_two)
        # cond_one_two_and_three = jnp.logical_and(cond_one_and_two, cond_three)
        return cond_one_and_two
    
    def action_policy(self, state: EnvState, params: EnvParams, side_idx: int, player_idx: int):
        # NOTE: Side idx pretty much defines the support
        # TODO: Deal with nao idx in the idle scenario
        # TODO: Where am I updating the support_policy? Can't update it inside this loop... Maybe at the beginning step?
        # ~~TODO: Double check this with the original function~~
        # TODO: Passing out support player param
        # TODO: Nao needs a slightly modified policy since it's policy can't be identical else Nao will follow supported player around 

        # self.support_policy.update_support(state)
        
        player_pos = state.player_state.player_positions[player_idx]
        nearest_bullet = self.get_nearest_bullet(state, params, side_idx, player_idx)
        nearest_enemy = self.get_nearest_enemy(state, params, side_idx, player_idx)
        can_shoot = self.can_shoot(state, params, player_idx)

        nearest_bullet_diff = nearest_bullet - player_pos

        # Agent is (in danger of being) hit if the x value of the nearest bullet is in there hitbox 
        hit = jnp.less_equal(
            jnp.abs(nearest_bullet_diff[params.x_idx]), 
            params.dynamics_consts.hit_range
        ) 

        # Action selection when agent cannot shoot 
        left_and_hit_if_cant_shoot = jnp.logical_and(
            hit, 
            jnp.logical_or(
                nearest_bullet[params.x_idx] >= 1 - (75 / params.render_consts.screen_width),
                nearest_bullet[params.x_idx] > player_pos[params.x_idx]
            )
        )
    
        right_and_hit_if_cant_shoot = jnp.logical_and(
            hit, 
            jnp.logical_or(
                nearest_bullet[params.x_idx] <= 55 / params.render_consts.screen_width, 
                nearest_bullet[params.x_idx] <= player_pos[params.x_idx]
            ) 
        )

        # Action selection when agent can shoot
        y_thresh_cond = jnp.less(
            player_pos[params.y_idx] - nearest_bullet[params.y_idx], 
            200 / params.render_consts.screen_height
        )

        left_bullet_danger = jnp.logical_and(
            -nearest_bullet_diff[params.x_idx] <= params.dynamics_consts.second_hit_range, 
            jnp.logical_and(
                nearest_bullet[params.x_idx] < player_pos[params.x_idx],
                y_thresh_cond
            )
        )
        
        right_bullet_danger = jnp.logical_and(
            nearest_bullet_diff[params.x_idx] <= params.dynamics_consts.second_hit_range,
            jnp.logical_and(
                nearest_bullet[params.x_idx] > player_pos[params.x_idx],
                y_thresh_cond)
        )

        left_and_approaching = jnp.logical_and(
            nearest_enemy[params.x_idx] < player_pos[params.x_idx], 
            ~left_bullet_danger
        )

        right_and_approaching = jnp.logical_and(
            nearest_enemy[params.x_idx] > player_pos[params.x_idx],
            ~right_bullet_danger
        )

        approaching = jnp.logical_or(
            left_and_approaching, 
            right_and_approaching
        )
        
        shoot_when_approaching = jnp.less_equal(
            jnp.abs(nearest_enemy[params.x_idx] - player_pos[params.x_idx]), 
            params.dynamics_consts.shooting_range
        )

        # If bullet is near and we’re not approaching, we used to forbid shooting.
        not_approaching_and_hit = jnp.logical_and(jnp.logical_not(approaching), hit)
        not_shoot_when_not_approaching = jnp.greater(
            jnp.abs(nearest_bullet_diff[params.x_idx]), 
            5 / params.render_consts.screen_width
        )

        not_shoot_when_not_approaching_and_hit = jnp.logical_and(
            not_approaching_and_hit, 
            not_shoot_when_not_approaching
        )

        def can_shoot_fn(_):
            left_val = jnp.logical_or(
                left_and_approaching, 
                jnp.logical_and(jnp.logical_not(approaching), left_and_hit_if_cant_shoot)
            )
            right_val = jnp.logical_or(
                right_and_approaching, 
                jnp.logical_and(jnp.logical_not(approaching), right_and_hit_if_cant_shoot)
            )
            shoot_val = jnp.logical_and(
                shoot_when_approaching,
                jnp.logical_not(not_shoot_when_not_approaching_and_hit),
            )
            return (left_val, right_val, shoot_val)
        
        def cant_shoot_fn(_):
            return (left_and_hit_if_cant_shoot, right_and_hit_if_cant_shoot, jnp.array(False))
        
        left, right, shoot = jax.lax.cond(
            can_shoot,
            can_shoot_fn,
            cant_shoot_fn,
            operand=None
        )

        # # Tie breaking: If you can shoot or move, move. 
        # shoot = jax.lax.cond(
        #     jnp.logical_and(shoot, jnp.logical_or(left, right)),
        #     lambda _: jnp.array(False),
        #     lambda _: shoot,
        #     operand=None
        # )
    
        # Tie breaking: For Shutter and Human, if they can shoot and move, set direction to false because on next frame they can move
        # TODO: Nao should use second nearest enemy in their implementation
        left = jax.lax.cond(jnp.logical_and(left, shoot),
                            lambda _: jnp.array(False),
                            lambda _: left,
                            None)
        
        right = jax.lax.cond(jnp.logical_and(right, shoot),
                             lambda _: jnp.array(False),
                             lambda _: right,
                             None)
        
        # Player can only do an action if that action is 1 and player is active (1)
        player_is_active = state.player_state.players_active[player_idx]
        left  = jnp.logical_and(left, player_is_active)
        right = jnp.logical_and(right, player_is_active)
        shoot = jnp.logical_and(shoot, player_is_active)
        nothing = (~left & ~right) & ~shoot
        # nothing = jnp.logical_and(jnp.logical_and(jnp.logical_not(left), jnp.logical_not(right)), jnp.logical_not(shoot))
        
        return jnp.array([left, right, shoot, nothing]).flatten()
    
class JxHumanRulesBasedPolicy(JxBaseActionPolicy):
    def __init__(self):
        super().__init__()

class JxShutterRulesBasedPolicy(JxBaseActionPolicy):
    def __init__(self):
        super().__init__()

class JxNaoRulesBasedPolicy(JxBaseActionPolicy):
    def __init__(self):
        super().__init__()

    def get_nearest_enemy_orig(self, state: EnvState, params: EnvParams, side_idx: int, player_idx: int):
        enemy_positions = state.enemy_state.enemy_positions[:, side_idx]  # (2, 5, 5)
        active_enemies = state.enemy_state.enemies_active[side_idx]
        active_enemies_broadcasted = jnp.broadcast_to(active_enemies, shape=enemy_positions.shape) # (2, 5, 5)

        # Mask based on whether enemy is active
        active_only_enemy_positions = jax.lax.select(active_enemies_broadcasted, 
                                                     enemy_positions,
                                                     jnp.full_like(a=enemy_positions, fill_value=jnp.inf, dtype=float))
        
        # Get distances 
        player_pos = state.player_state.player_positions[player_idx] # (2,)
        x_dists = jnp.abs(active_only_enemy_positions[params.x_idx] - player_pos[params.x_idx]) # [2, 5, 5]
        y_dists = jnp.abs(active_only_enemy_positions[params.y_idx] - player_pos[params.y_idx])
        min_y_dist = jnp.min(y_dists)

        # Mask all y dists not equal to min y dist
        equal_min_y_mask = jnp.equal(y_dists, min_y_dist)

        # Only x dists where y is equal to min y dist
        x_dists_where_y_dist_is_min = jax.lax.select(equal_min_y_mask, 
                                                     x_dists, 
                                                     jnp.full_like(a=x_dists, fill_value=jnp.inf, dtype=float))
                
        # Mask all x dists where y isn't equal to min y dist
        x_dists_where_y_dist_is_min_mask = jnp.equal(x_dists, jnp.min(x_dists_where_y_dist_is_min))

        # 1 everywhere y is minimal and x is minimal where y is minimal, 0 otherwise 
        final_mask = jnp.logical_and(equal_min_y_mask, x_dists_where_y_dist_is_min_mask)
        nearest_enemy_idx = jnp.argmax(final_mask)

        # Construct nearest enemy. Flatten because no axis specified in argmax
        nearest_enemy_x = active_only_enemy_positions[params.x_idx].flatten()[nearest_enemy_idx]
        nearest_enemy_y = active_only_enemy_positions[params.y_idx].flatten()[nearest_enemy_idx]
        nearest_enemy_position = jnp.array([nearest_enemy_x, nearest_enemy_y])
        return nearest_enemy_position, nearest_enemy_idx
    
    def get_nearest_enemy(self, state: EnvState, params: EnvParams, side_idx: int, player_idx: int):
        _, nearest_enemy_idx = self.get_nearest_enemy_orig(state, params, side_idx, side_idx)
        enemy_positions = state.enemy_state.enemy_positions
        # Set the enemy position of nearest enemy supp in enemy positions to inf 
        enemy_positions_flat = enemy_positions[:, state.sides[params.nao_idx]].flatten()
        enemy_positions_flat = enemy_positions_flat.at[nearest_enemy_idx].set(jnp.inf)
        new_enemy_positions = enemy_positions.at[:, state.sides[params.nao_idx]].set(enemy_positions_flat.reshape(2, 5, 5))
        n_state = state.replace(
            enemy_state=state.enemy_state.replace(
                enemy_positions=new_enemy_positions
            )
        )
        nearest_enemy_position, _ = self.get_nearest_enemy_orig(n_state, params, side_idx, player_idx)
        return nearest_enemy_position

    def nao_policy(self, state: EnvState, params: EnvParams, side_idx, pos_idx):
        nao_or_player_support = state.sides[params.nao_idx] == params.nao_idx

        def idle_policy(state: EnvState, params: EnvParams, tmp1, tmp2):
            offset = .005
            nao_x = state.player_state.player_positions[params.nao_idx, params.x_idx]
            move_left = (nao_x > .5 + offset)
            move_right = (nao_x < .5 - offset)
            move = move_left | move_right
            
            def inner_cond(move_left):
                return jax.lax.cond(
                    move_left,
                    lambda _: jnp.array([True, False, False, False]),
                    lambda _: jnp.array([False, True, False, False]),
                    None
                )
            
            return jax.lax.cond(
                move,
                lambda _: inner_cond(move_left),
                lambda _: jnp.array([False, False, False, True]),
                move_left
            )

        return jax.lax.cond(
            nao_or_player_support,
            lambda _: idle_policy(state, params, side_idx, pos_idx),
            lambda _: self.action_policy(state, params, side_idx, pos_idx),
            None
        )

# TODO: Pretty sure this doesn't have to be a dataclass lol
@struct.dataclass
class MultiAgentPolicy:
    
    policy_funcs: Tuple # = struct.field(pytree_node=False)

    @functools.partial(jax.jit, static_argnums=(0,))
    def joint_policy(self, state: EnvState, params: EnvParams):
        # TODO: update Nao's support here
        vmap_functions = jax.vmap(lambda i, inds: jax.lax.switch(i, self.policy_funcs, state, params, *inds))
        return vmap_functions(state.player_idxs, (state.sides, state.player_idxs))
    
    @classmethod
    def reset(cls):
        policy_funcs = (JxHumanRulesBasedPolicy().action_policy, 
                        JxShutterRulesBasedPolicy().action_policy, 
                        JxNaoRulesBasedPolicy().nao_policy)

        return cls(policy_funcs=policy_funcs)
    