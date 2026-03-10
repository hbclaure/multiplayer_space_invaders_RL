import functools
from typing import Tuple
import jax
import jax.numpy as jnp
from gymnax.environments import environment
from env.jx_env_state_params import *
from jax.experimental import checkify

class SpaceInvadersJax(environment.Environment[EnvState, EnvParams]):
    # ~~TODO: Bullet logic for generating new bullets per entity (player or enemy) --> something with rolling most likely~~
        # Just prespecified maximum number of bullets instead. 
    # ~~TODO: Policies for Nao, Human, and Shutter~~
    # ~~TODO: Nao, Shutter and Human shooting 3 bullets back to back~~
    # ~~TODO: Where am I putting Nao, Human, and Shutter policies?~~
    # ~~TODO: Step func~~
    # ~~TODO: Collision logic with enemy bullets and players needs to be fixed~~ 
    # ~~TODO: Ensure deaths aren't being shared between enemies~~
    # ~~TODO: Ensure that deaths aren't being shared between players~~
    # ~~TODO: Enemy can shoot logic needs to be fixed~~
    # ~~TODO: Stop enemies from double/triple shooting~~ 
    # ~~TODO: Fix respawn logic for players and enemies~~
    # ~~TODO: Don't let enemies shoot if no players are on their side~~
    # ~~TODO: Nao following player they're supporting~~
    # ~~TODO: Nao updates support properly~~
    # ~~TODO: Saving and loading trajectories~~
    # ~~TODO: Finish up and optimize rendering a bit (less loops)~~
    # ~~TODO: Keep track of which side nao shoots on~~
    # TODO: Rules-based policy tweaking. 
    # TODO: Cooperative vs Competitive Mode 
    # ~~TODO: Proper tracking of all the relevant variables in StepInfo~~
    # TODO: Match naming conventions with original environment
    # TODO: Don't let players shoot if there are no enemies on their side
        # NOTE: Pretty low priority fix -- doubt there's ever a team all enemies are gone 

    # TODO: Rendering 
    # TODO: Unit tests 
    # TODO: Comment all code
    # ~~TODO: Make sure replace functions use same names as tho in envparams~~
    # ~~TODO: Are broadcast_tos fine?~~
    #~~ TODO: How to save all the state relevant information in json files similar to how we have it currently~~
        # TODO: Pickle instead is simpler 
    # ~~TODO: Rewrite functions using enemy positions, enemies active, and bullets to account for shape change~~
    # TODO: Properly update the rewards between steps 
    # ~~TODO: Check to make sure all functions are pure (i.e., depend solely on variables passed into the function)~~
    # TODO: Probably need to downgrade Jax version to ensure jax while loop is fast 
    # ~~TODO: Dynamics~~

    # ~~TODO: Replace wheres and selects with jax.lax.cond where relevant~~ 
    # ~~TODO: Keep as many things the same shape as possible~~
    # ~~TODO: Get rid of checks once everything passes~~
    # ~~TODO: Update all functions to uphold to new conventions for indexing~~
    # ~~TODO: Is it okay that state.replace has no func or property signature?~~
    # ~~TODO: when broadcasting, do i need to expand first?~~
    # ~~TODO: Handle player location respawns~~
    # TODO: Finish all empty functions
    # ~~TODO: Shutter teleporting?~~
    # ~~TODO: Bullet collision checking for everyone~~
    # ~~TODO: Every bottom row agent shooting~~

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step_env(self, key, state: EnvState, actions: jax.Array, params: EnvParams):
        # Implement the logic for stepping the environment
        # TODO: Get rid of this before jitting and going to step
        key, key_reset = jax.random.split(key)
        # ~~TODO: Verify below runs~~
        # TODO: Verify consistency with original 
        state = self._action_handler(actions, state, params)
        # ~~TODO: Verify below runs~~
        # TODO: Verify consistency with original 
        state = self._enemy_shot_handler(key, state, params)
        # ~~TODO: Verify below runs~~
        # TODO: Verify consistency with original 
        state = self._bullet_threat_check(state, params)
        # ~~TODO: Verify below runs~~
        # ~~TODO: Verify consistency with original~~
        state = self._update_player_bullet_positions(state, params)
        state = self._update_enemy_bullet_positions(state, params)
        # ~~TODO: Verify below runs~~
        # TODO: Verify consistency with original 
        state = self._check_hb_collisions_on_enemies(state, params)
        state = self._check_eb_collisions_on_players(state, params)
        # ~~TODO: Verify below runs~~
        # TODO: Verify consistency with original
        state = self._player_respawn_handler(state, params)
        # ~~TODO: Verify below runs~~
        # TODO: Verify consistency with original
        state = self._enemy_respawn_handler(state, params)
        # ~~TODO: Verify below runs~~
        # TODO: Verify consistency with original
        state = self._score_handler(state, params)
        state = self._update_time(state, params) 
        state = self._update_sides(state, params)
        # TODO: whatever else I need to keep track of 
        return 0, state, 0, 0, 0

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset_env(
        self, key: jax.Array, params: EnvParams
    ):
        state = EnvState.reset()
        #TODO: return get_obs(state) too 
        return 0, state
    
    def _update_sides(self, state: EnvState, params: EnvParams):
        # TODO: Maybe these should return indexes instead? 
        def idle_support(_, params: EnvParams):
            return jnp.array([params.human_idx, params.shutter_idx, params.nao_idx])
        
        def only_human(_, params: EnvParams):
            return jnp.array([params.human_idx, params.shutter_idx, params.human_idx])
        
        def only_shutter(_, params: EnvParams):
            return jnp.array([params.human_idx, params.shutter_idx, params.shutter_idx])
        
        def equalize_scores(state: EnvState, params: EnvParams):
            human_score = state.score_state.scores[params.human_idx] + state.score_state.scores[params.nao_for_human_score_idx]
            shutter_score = state.score_state.scores[params.shutter_idx] + state.score_state.scores[params.nao_for_shutter_score_idx]
            score_diff = human_score - shutter_score
            score_diff_gt_thresh = jnp.abs(score_diff) >= params.policy_consts.score_difference_threshold
            supp_human = score_diff_gt_thresh & (score_diff > 0)
            supp_shutter = score_diff_gt_thresh & (score_diff < 0)

            def inner_cond(supp_human):
                return jax.lax.cond(
                    supp_human,
                    lambda _: params.human_idx,
                    lambda _: params.shutter_idx,
                    None
                )
            
            new_side = jax.lax.cond(
                supp_human | supp_shutter, 
                lambda _: inner_cond(supp_human),  
                lambda _: state.sides[params.nao_idx], 
                None
            )
            return jnp.array([params.human_idx, params.shutter_idx, new_side])
        
        def equalize_scores_neutral(state: EnvState, params: EnvParams):
            human_score = state.score_state.scores[params.human_idx] + state.score_state.scores[params.nao_for_human_score_idx]
            shutter_score = state.score_state.scores[params.shutter_idx] + state.score_state.scores[params.nao_for_shutter_score_idx]
            score_diff = human_score - shutter_score
            score_diff_gt_thresh = jnp.abs(score_diff) >= params.policy_consts.score_difference_threshold
            supp_human = score_diff_gt_thresh & (score_diff > 0)
            supp_shutter = score_diff_gt_thresh & (score_diff < 0)

            def inner_cond(supp_human):
                return jax.lax.cond(
                    supp_human,
                    lambda _: params.human_idx,
                    lambda _: params.shutter_idx,
                    None
                )
            
            new_side = jax.lax.cond(
                supp_human | supp_shutter, 
                lambda _: inner_cond(supp_human),  
                lambda _: params.nao_idx, 
                None
            )
            return jnp.array([params.human_idx, params.shutter_idx, new_side])
        
        branches = [idle_support, only_human, only_shutter, equalize_scores, equalize_scores_neutral]
        new_sides = jax.lax.switch(state.player_state.nao_supp_func_idx, branches, state, params)
        return state.replace(
            sides=new_sides
        )

    def _action_handler(self, actions: jax.Array, state: EnvState, params: EnvParams):
        # ~~TODO: Figure out how I want to do this~~ 
        # TODO: Put movement amounts in params under dynamics_consts
        # Action is shape (n_players, n_actions)
        # ~~TODO: How are returns being done here?~~

        def move_left(curr_state: EnvState, player_idx: int):
            new_player_position_x = jnp.clip(
                curr_state.player_state.player_positions[player_idx, params.x_idx] - 0.00625, 
                params.dynamics_consts.move_left_bounds[player_idx], 
                1
            )
            new_player_positions = curr_state.player_state.player_positions.at[player_idx, params.x_idx].set(new_player_position_x)

            return curr_state.replace(
                player_state=curr_state.player_state.replace(player_positions=new_player_positions)
            )
        
        def move_right(curr_state: EnvState, player_idx: int):
            new_player_position_x = jnp.clip(
                curr_state.player_state.player_positions[player_idx, params.x_idx] + 0.00625, 
                0, 
                params.dynamics_consts.move_right_bounds[player_idx]
            )
            new_player_positions = curr_state.player_state.player_positions.at[player_idx, params.x_idx].set(new_player_position_x)

            return curr_state.replace(
                player_state=curr_state.player_state.replace(player_positions=new_player_positions)
            )
            
        def shoot(curr_state: EnvState, player_idx: int):
            # ~~TODO: Second half of bullet handling logic is going here for now. Will eventually migrate it to policies class instead~~
            # ~~TODO: Verify that I also get rid of bullet indices properly~~
            # TODO: Triple check that I'm updating bullets on collisions and oobs correctly
            # TODO: Why is bullet speed in shoot diff??
            # TODO: Keep track of reward const for shooting here

            # Get lowest inactive bullet (value of 0) 
            new_bullet_idx = jnp.argmin(curr_state.bullet_state.player_bullet_actives[player_idx, :])

            # Lowest inactive bullet is now active
            new_player_bullet_actives = curr_state.bullet_state.player_bullet_actives.at[player_idx, new_bullet_idx].set(1) 

            # Spawn bullet 
            new_player_bullet_positions = curr_state.bullet_state.player_bullet_positions.at[player_idx, new_bullet_idx, :].set(curr_state.player_state.player_positions[player_idx, :])
            new_player_bullet_positions = new_player_bullet_positions.at[player_idx, new_bullet_idx, params.y_idx].add(params.dynamics_consts.player_shoot_bullet_speed)

            # ~~Not sure if I need to add + 0, but it's safer this way~~
            new_player_last_shot_frame = curr_state.history.player_last_shot_frame.at[player_idx].set(curr_state.time_state.frame_num)
            new_player_shoot_rewards = jnp.zeros(3, dtype=int)
            new_player_shoot_rewards = new_player_shoot_rewards.at[player_idx].set(params.reward_consts.player_shoot_reward)

            return curr_state.replace(
                bullet_state=curr_state.bullet_state.replace(player_bullet_actives=new_player_bullet_actives, 
                                                             player_bullet_positions=new_player_bullet_positions),
                history=curr_state.history.replace(player_last_shot_frame=new_player_last_shot_frame.astype(int)),
                step_info=curr_state.step_info.replace(player_shoot_rewards=new_player_shoot_rewards)
            )

        def nothing(curr_state, _):
            return curr_state
        
        def apply_action(state_action_pidx: Tuple):
            curr_state, action, player_idx = state_action_pidx
            index = jnp.argmax(action)
            branches = [move_left, move_right, shoot, nothing]
            return jax.lax.switch(index, branches, curr_state, player_idx)    
        
        def policy_step(state_idx, tmp):
            curr_state, idx = state_idx
            next_state = apply_action((curr_state, actions[idx], player_idxs[idx]))  
            carry = (next_state, idx+1)
            return carry, next_state
                
        # TODO: Unit test to make sure policy step is invariant to action order
        
        player_idxs = jnp.array([params.human_idx, params.shutter_idx, params.nao_idx])
        scan_output,  _ = jax.lax.scan(f=policy_step, init=(state, 0), xs=(), length=len(state.player_idxs))
        final_state, _ = scan_output
        return final_state.replace(
            step_info=final_state.step_info.replace(actions_taken=actions.astype(int))
        )

    def _enemy_shot_handler(self, key, state: EnvState, params: EnvParams):
        # ~~TODO: finish this~~ 
        # TODO: I think its better to define enemy_last_shot_frame on enemies individually, or at least per side, than globally.
        #       Adjust accordingly.
        # TODO: Need to constrain the number of bullets allowed to be shot at the same time. 
        #       can do so by allowing only the lowest active enemy to shoot and constraining on 
        #       max bullets possible 
        # params.fps = 60
        # TODO: Modify to account for enemy bullet actives being shape (2, 2, 5) --> (n_bullets, side_bool, n_cols)
        # TODO: Modify to account for enemy bullet positions being shape (2, 2, 2, 5) --> (n_bullets, n_dims, side_bool, n_cols)

        # An enemy can shoot if the time since the last shot is more than a random delay ...
        random_delay_frames = jax.random.uniform(
            key=key, 
            minval=.7 * params.dynamics_consts.fps, 
            maxval=params.dynamics_consts.fps, 
            shape=()
        )       

        can_shoot = state.time_state.frame_num - state.history.enemy_last_shot_frame > random_delay_frames 
        
        # and the number of total bullets in a column is less than the max ...
        num_bullets_per_col = jnp.sum(state.bullet_state.enemy_bullet_actives, axis=[0, 2]) # [side_bool, n_col] --> [2, 5]
        num_bullets_per_col_expanded = num_bullets_per_col[:, None, :]  # [side_bool, None, n_col]
        num_bullets_per_col_broadcasted = jnp.broadcast_to(
            num_bullets_per_col_expanded, 
            shape=state.enemy_state.enemies_active.shape
        ) # [side_bool, n_row, n_col] --> [2, 5, 5]

        can_shoot = jnp.logical_and(
            can_shoot,
            num_bullets_per_col_broadcasted < params.state_consts.max_num_enemy_bullets_per_column
        )

        # and there are players alive ...
        player_pos = state.player_state.player_positions
        players_act = state.player_state.players_active

        players_active_on_ls = jnp.logical_or(
            (player_pos[params.human_idx, params.x_idx] <= .5) & (players_act[params.human_idx]),
            (player_pos[params.nao_idx, params.x_idx] <= .5) & (players_act[params.nao_idx])
        )

        players_active_on_rs = jnp.logical_or(
            (player_pos[params.shutter_idx, params.x_idx] > .5) & (players_act[params.shutter_idx]),
            (player_pos[params.nao_idx, params.x_idx] > .5) & (players_act[params.nao_idx])
        )

        player_on_side = jnp.array([players_active_on_ls, players_active_on_rs])

        can_shoot = can_shoot * player_on_side[:, None, None]
        
        # and the enemy is active ...
            # NOTE: Only valid if enemies_active[:, 0, :] is the row closest to players, which by construction it is. 
        can_shoot = jnp.logical_and(can_shoot, state.enemy_state.enemies_active).astype(int)

        # and the enemy is the lowest in their column 
        lowest_can_shoot = jnp.equal(jnp.cumsum(can_shoot, axis=1), 1) # True for the lowest active enemy in each column that can shoot
        key1, key2 = jax.random.split(key)
        keys = jnp.array([key1, key2])
            # reduce along rows to see if col on side has a lowest enemy that can shoot 
        has_lowest_can_shoot = jnp.any(lowest_can_shoot, axis=1) # [side_bool, n_cols] --> [2, 5]
            # randomly sample the lowest enemy in a column that can shoot 
        def sample_col(k, w):
            valid_cols = jax.random.choice(
                key=k,
                a=jnp.arange(w.shape[0]),
                shape=(),
                p=w / jnp.sum(w)
            )
            return valid_cols
            
        valid_cols = jax.vmap(sample_col)(keys, has_lowest_can_shoot)
        col_mask = jnp.zeros_like(has_lowest_can_shoot)
        col_mask = col_mask.at[jnp.arange(2), valid_cols].set(1)

        can_shoot = lowest_can_shoot * col_mask[:, None, :]  # [2, 5, 5]
        
        # If an enemy can shoot, they shoot from their lowest active bullet
        inactive_bullets = jnp.logical_not(state.bullet_state.enemy_bullet_actives)  # [2, 2, 5, 5]
        next_available_bullet = jnp.equal(jnp.cumsum(inactive_bullets, axis=0), 1) # cumsum will be 1 for the first  across rows 

        can_shoot = jnp.broadcast_to(can_shoot[None, ...], shape=next_available_bullet.shape) # (2, 2, 5, 5)
        can_shoot = jax.lax.select(
            can_shoot.astype(bool),
            next_available_bullet,
            jnp.zeros_like(next_available_bullet).astype(dtype=bool)
        )
            # a bullet is active if an enemy can shoot from it 
        new_enemy_bullet_actives = jnp.logical_or(
            state.bullet_state.enemy_bullet_actives,
            can_shoot
        )

        # Expand and broadcast to prepare to spawn bullet at enemy who can shoot
        can_shoot = jnp.broadcast_to(
            can_shoot[:, None, ...], 
            state.bullet_state.enemy_bullet_positions.shape
        ) # [2, 2, 2, 5, 5]

        enemy_positions_broadcasted = jnp.broadcast_to(
            state.enemy_state.enemy_positions[None, ...], 
            state.bullet_state.enemy_bullet_positions.shape
        )

        enemy_positions_broadcasted = enemy_positions_broadcasted.at[:, params.y_idx, ...].add(params.dynamics_consts.enemy_bullet_velocity)
            # spawn bullet a bit ahead of the enemy who can shoot
        new_enemy_bullet_positions = jax.lax.select(
            can_shoot,
            enemy_positions_broadcasted,
            state.bullet_state.enemy_bullet_positions
        )

        # Reset shoot counter if any bullets were spawned
        any_spawned = jnp.any(can_shoot)
        new_enemy_last_shot_frame = jax.lax.cond(
            any_spawned, 
            lambda _: state.time_state.frame_num, 
            lambda _: state.history.enemy_last_shot_frame, 
            None
        )

        return state.replace(
            bullet_state=state.bullet_state.replace(
                enemy_bullet_positions=new_enemy_bullet_positions,
                enemy_bullet_actives=new_enemy_bullet_actives.astype(dtype=int),
            ),
            history=state.history.replace(
                enemy_last_shot_frame=new_enemy_last_shot_frame
            )
        )
    
    def _update_player_bullet_positions(self, state: EnvState, params: EnvParams):
        # Update the y values of the bullets 
        new_player_bullet_ys = state.bullet_state.player_bullet_positions[:, :, params.y_idx] + params.dynamics_consts.human_bullet_velocity

        # Bullets are out of bounds if they're greater than 1 or less than 0
        # NOTE: Bullets that are inactive will always count as out of bounds since they are inf 
        players_gt_ub = jnp.greater(new_player_bullet_ys, 1.) 
        players_lt_lb = jnp.less(new_player_bullet_ys, 0.) 

        player_bullet_oob = jnp.logical_or(players_gt_ub, players_lt_lb) 
        
        # Any bullet that is not oob is active 
        new_player_bullet_actives = jnp.logical_not(player_bullet_oob)

        # If a bullet is oob, its x and y values are now inf. Otherwise, it retains the current x value and updated y value
        new_player_bullet_xs = jax.lax.select(
            player_bullet_oob, 
            jnp.full_like(player_bullet_oob, jnp.inf, dtype=float),
            state.bullet_state.player_bullet_positions[:, :, params.x_idx]
        ) 
        
        new_player_bullet_ys = jax.lax.select(
            player_bullet_oob, 
            jnp.full_like(player_bullet_oob, jnp.inf, dtype=float),
            new_player_bullet_ys, 
        ) 
        
        new_player_bullet_positions = jnp.stack([new_player_bullet_xs, new_player_bullet_ys], axis=2) # (3, 4, 2)

        return state.replace(
            bullet_state=state.bullet_state.replace(player_bullet_positions=new_player_bullet_positions,
                                                    player_bullet_actives=new_player_bullet_actives.astype(dtype=int))
        )

    
    def _update_enemy_bullet_positions(self, state: EnvState, params: EnvParams): 
         
        new_enemy_bullet_ys = state.bullet_state.enemy_bullet_positions[:, params.y_idx] + params.dynamics_consts.enemy_bullet_velocity  
        
        # Bullets are out of bounds if they're greater than 1 or less than 0
        # NOTE: Bullets that are inactive will always count as out of bounds since they are inf 
        enemies_gt_ub = jnp.greater(new_enemy_bullet_ys, 1.) 
        enemies_lt_lb = jnp.less(new_enemy_bullet_ys, 0.) 
        enemy_bullet_oob = jnp.logical_or(enemies_gt_ub, enemies_lt_lb) 

        # Any bullet that is not oob is active 
        new_enemy_bullet_actives = jnp.logical_not(enemy_bullet_oob)
       
        # If a bullet is oob, its x and y values are now inf. Otherwise, it retains the current x value and updated y value
        new_enemy_bullet_xs = jax.lax.select(
            enemy_bullet_oob, 
            jnp.full_like(enemy_bullet_oob, jnp.inf, dtype=float),
            state.bullet_state.enemy_bullet_positions[:, params.x_idx], 
        ) 

        new_enemy_bullet_ys = jax.lax.select(
            enemy_bullet_oob, 
            jnp.full_like(enemy_bullet_oob, jnp.inf, dtype=float),
            new_enemy_bullet_ys, 
        ) 
        # reconstruct the bullet positions now that they've been updated
        new_enemy_bullet_positions = jnp.stack([new_enemy_bullet_xs, new_enemy_bullet_ys], axis=1) # (2, 2, 2, 5, 5)

        return state.replace(
            bullet_state=state.bullet_state.replace(enemy_bullet_positions=new_enemy_bullet_positions,
                                                    enemy_bullet_actives=new_enemy_bullet_actives.astype(dtype=int))
        )

    
    def _bullet_threat_check(self, state: EnvState, params: EnvParams): 

        left_edges = state.player_state.player_positions[:, params.x_idx] - (params.render_consts.ship_width / 2)
        right_edges = state.player_state.player_positions[:, params.x_idx] + (params.render_consts.ship_width / 2)
        upper_edges = state.player_state.player_positions[:, params.y_idx] + (params.render_consts.ship_height / 2) # (3, )
        
        # Clip edges such that they're between 0 and 1 for the left edges and right edges respectively 
        left_edges = jnp.clip(left_edges, 0., 1.) # [:, None]
        right_edges = jnp.clip(right_edges, 0., 1.)# [:, None]

        left_edges = left_edges[:, None, None, None, None]  # (3, 1, 1, 1, 1)
        right_edges = right_edges[:, None, None, None, None]

        

        distances = upper_edges[:, None, None, None, None] - state.bullet_state.enemy_bullet_positions[:, params.y_idx, ...] # (3, 2, 2, 5, 5)
 
        # A bullet is a threat to a player if it's above the player, active, and inbetween the player's hori. hitbox 
        # (3, 2, 2, 5, 5) 
        between_cond = jnp.logical_and(
            left_edges <= state.bullet_state.enemy_bullet_positions[:, params.x_idx, ...], # (2, 2, 5, 5)
            right_edges >= state.bullet_state.enemy_bullet_positions[:, params.x_idx, ...]
        )

        threat_cond = jnp.logical_and(
            between_cond, 
            distances > 0.
        )

        new_bullet_threat_binary = jnp.logical_and(
            threat_cond, 
            state.bullet_state.enemy_bullet_actives
        ) # (3, 2, 2, 5, 5)

        # Consider only the distances of threatening bullets 
        masked_distances = jax.lax.select(
            new_bullet_threat_binary, 
            distances, 
            jnp.full_like(distances, jnp.inf, dtype=float)
        ) # (3, 2, 2, 5, 5)

        # Masked distances in terms of the number of frames to collision 
        new_frames_until_collison = jnp.abs(jnp.min(masked_distances, axis=[1, 2, 3, 4]) / params.dynamics_consts.enemy_bullet_velocity)

        return state.replace(
            bullet_state=state.bullet_state.replace(bullet_threat_binary=new_bullet_threat_binary.astype(dtype=int),
                                                    frames_until_collision=new_frames_until_collison)
        )
    
    def _check_hb_collisions_on_enemies(self, state: EnvState, params: EnvParams):  # Check if any bullets have collided with enemies 
        # TODO: What step info things need to be updated here

        # Construct the enemies hitboxes as rectangles parameterized by left and top coordinates, width, and height 
        enemy_lefts = (state.enemy_state.enemy_positions[params.x_idx, ...] * params.render_consts.screen_width) - (params.render_consts.ship_width / 2)  # (2, 5, 5)
        enemy_bots = (state.enemy_state.enemy_positions[params.y_idx, ...] * params.render_consts.screen_height) - (params.render_consts.ship_height / 2)  # (2, 5, 5)
        enemy_rights = enemy_lefts + params.render_consts.ship_width
        enemy_tops = enemy_bots + params.render_consts.ship_height
        # enemy_width = params.render_consts.ship_width * params.render_consts.screen_width
        # enemy_height = params.render_consts.ship_height * params.render_consts.screen_height

        # Construct the bullet from the players individually, each with hitboxes as rectangles parameterized by left and top coordinates, width, and height 
        bullet_lefts = (state.bullet_state.player_bullet_positions[:, :, params.x_idx] * params.render_consts.screen_width) - (params.render_consts.bullet_width / 2) # (3, 4)
        bullet_tops = (state.bullet_state.player_bullet_positions[:, :, params.y_idx]  * params.render_consts.screen_height) + (params.render_consts.bullet_height / 2) # (3, 4)
        
        # Check intersections of the bullets and enemies assuming both are rectangles

        # All intersects are (3, 4, 2, 5, 5)

        bullet_lefts = bullet_lefts[:, :, None, None, None]  # (3, 4, 1, 1, 1)
        bullet_rights = bullet_lefts + params.render_consts.bullet_width
        bullet_tops = bullet_tops[:, :, None, None, None] 
        bullet_bots = bullet_tops - params.render_consts.bullet_height

        no_overlap_x = (bullet_lefts >= enemy_rights) | (enemy_lefts >= bullet_rights)
        no_overlap_y = (bullet_bots >= enemy_tops) | (enemy_bots >= bullet_tops)
        
        # Intersecting if not to the left or above
        player_bullet_enemy_intersections = jnp.logical_not(jnp.logical_or(no_overlap_x, no_overlap_y)) # [3, 4, 2, 5, 5]
        
        # Reduce over the player bullets to see if an enemy has been hit
        enemy_hit_by_any_player = jnp.any(player_bullet_enemy_intersections, axis=[0, 1]) # [2, 5, 5]
        enemy_hit_by_any_player_broadcasted = jnp.broadcast_to(
            enemy_hit_by_any_player[None, ...], # [2, 1, 5, 5]
            state.enemy_state.enemy_positions.shape
        ) # [2, 2, 5, 5]
        
        # If an enemy was hit, it's no longer active. Else, select whether it's active right now.   
        new_enemies_active = jax.lax.select(
            enemy_hit_by_any_player,
            jnp.zeros_like(state.enemy_state.enemies_active),
            state.enemy_state.enemies_active
        )
        
        # If an enemy was hit, it's position is inf. Else, select its current position. 
        new_enemy_positions = jax.lax.select(
            enemy_hit_by_any_player_broadcasted,
            jnp.full_like(state.enemy_state.enemy_positions, jnp.inf),
            state.enemy_state.enemy_positions
        )
        
        # If an enemy was hit, it's frames since death is 0. Else, select its current frames since death value.
        new_enemy_frames_since_death = jax.lax.select(
            enemy_hit_by_any_player, 
            jnp.zeros_like(state.history.enemy_frames_since_death),
            state.history.enemy_frames_since_death
        )
        
        # Reduce over the enemy positions to see if a bullet has hit an enemy
        player_bullet_hit_enemy = jnp.any(player_bullet_enemy_intersections, axis=[2, 3, 4]) # (3, 4)
        player_bullet_hit_enemy_broadcasted = jnp.broadcast_to(
            player_bullet_hit_enemy[..., None],
            state.bullet_state.player_bullet_positions.shape
        )

        # If a bullet hit something, deactivate it. Else, select whether its currently active or not. 
        new_player_bullet_actives = jax.lax.select(
            player_bullet_hit_enemy, 
            jnp.zeros_like(state.bullet_state.player_bullet_actives), 
            state.bullet_state.player_bullet_actives
        ) 
        
        # If a bullet hit something, it's position is inf. Else, select its current position. 
        new_player_bullet_positions = jax.lax.select(
            player_bullet_hit_enemy_broadcasted,
            jnp.full_like(state.bullet_state.player_bullet_positions, jnp.inf),
            state.bullet_state.player_bullet_positions
        )

        # Compute hitting enemy scores 
        # By convention, if Nao's bullet hits the same opponent as the Human or Shutter in the same frame, we count that hit for Human/Shutter 
        enemy_hit_by_player = jnp.any(player_bullet_enemy_intersections, axis=1) 
        enemy_hit_by_human = enemy_hit_by_player[params.human_idx, ...]
        enemy_hit_by_shutter = enemy_hit_by_player[params.shutter_idx, ...]
        enemy_hit_by_nao = enemy_hit_by_player[params.nao_idx, ...]
        
        enemy_hit_by_human_and_nao = jnp.logical_and(enemy_hit_by_human, enemy_hit_by_nao) 
        enemy_hit_by_shutter_and_nao = jnp.logical_and(enemy_hit_by_shutter, enemy_hit_by_nao) 

        enemy_hit_by_nao = jax.lax.select(enemy_hit_by_human_and_nao,
                                          jnp.zeros_like(enemy_hit_by_human_and_nao),
                                          enemy_hit_by_nao)

        enemy_hit_by_nao = jax.lax.select(enemy_hit_by_shutter_and_nao,
                                          jnp.zeros_like(enemy_hit_by_shutter_and_nao),
                                          enemy_hit_by_nao)
        
        num_enemies_hit_by_human = jnp.count_nonzero(enemy_hit_by_human)
        num_enemies_hit_by_shutter = jnp.count_nonzero(enemy_hit_by_shutter)
        num_enemies_hit_by_nao = jnp.count_nonzero(enemy_hit_by_nao)

        nao_side = state.player_state.player_positions[params.nao_idx, params.x_idx] < .5 # 1 if nao on human side

        new_enemy_elimination_rewards = jnp.array([
            params.reward_consts.enemy_elimination_reward * num_enemies_hit_by_human,
            params.reward_consts.enemy_elimination_reward * num_enemies_hit_by_shutter,
            params.reward_consts.enemy_elimination_reward * num_enemies_hit_by_nao
        ])

        new_score_changes = jnp.array([
            params.score_consts.bonus_for_hitting_enemy * num_enemies_hit_by_human,
            params.score_consts.bonus_for_hitting_enemy * num_enemies_hit_by_shutter,
            params.score_consts.bonus_for_hitting_enemy * num_enemies_hit_by_nao,
            params.score_consts.bonus_for_hitting_enemy * num_enemies_hit_by_nao * nao_side,
            params.score_consts.bonus_for_hitting_enemy * num_enemies_hit_by_nao * (1 - nao_side),
        ])

        new_scores = state.score_state.scores + new_score_changes

        # ~~TODO: Modify to account for nao helping...~~
        h_gt_s = jnp.greater(
            state.score_state.scores[params.human_idx] + new_enemy_elimination_rewards[params.nao_idx] * nao_side,
            state.score_state.scores[params.shutter_idx] + new_enemy_elimination_rewards[params.nao_idx] * (1 - nao_side)
        )
        
        # TODO: add extra logic for taking into consideration whether scores equalized
        new_player_taking_score_lead_rewards = jax.lax.cond(h_gt_s, 
                                                            lambda _: jnp.array([params.reward_consts.player_taking_score_lead_reward, -params.reward_consts.player_taking_score_lead_reward, 0]),
                                                            lambda _: jnp.array([-params.reward_consts.player_taking_score_lead_reward, params.reward_consts.player_taking_score_lead_reward, 0]),
                                                            None)
        
        # ~~TODO: Score tracking~~
        # ~~TODO: Get rid of bullets that collided~~
        # ~~TODO: Update enemy frames~~
        # TODO: Lead change reward 
        # TODO: Update score state here 

        return state.replace(
            enemy_state=state.enemy_state.replace(enemy_positions=new_enemy_positions,
                                                   enemies_active=new_enemies_active.astype(dtype=int)),
            history=state.history.replace(enemy_frames_since_death=new_enemy_frames_since_death),
            bullet_state=state.bullet_state.replace(player_bullet_actives=new_player_bullet_actives,
                                                    player_bullet_positions=new_player_bullet_positions),
            step_info=state.step_info.replace(enemy_elimination_rewards=new_enemy_elimination_rewards.astype(int),
                                              player_taking_score_lead_rewards=new_player_taking_score_lead_rewards,
                                              score_changes=new_score_changes[:3]),
            score_state=state.score_state.replace(scores=new_scores)
        )
    
    def _check_eb_collisions_on_players(self, state: EnvState, params: EnvParams):
        # TODO: Agent powerup checking via invincibility 
        # NOTE: Agent power up in state
        # TODO: Why is checking for collisions so different than checking for threats in terms of parameters? 
        # TODO: Properly despawn bullets after collision

        player_lefts = (state.player_state.player_positions[:, params.x_idx] * params.render_consts.screen_width) - (params.render_consts.ship_width / 2)  # (3,)
        player_bots = (state.player_state.player_positions[:, params.y_idx] * params.render_consts.screen_height) - (params.render_consts.ship_height / 2)  # (3,)

        player_lefts = player_lefts[:, None, None, None, None]  # (3, 1, 1, 1, 1)
        player_bots = player_bots[:, None, None, None, None]
        player_rights = player_lefts + params.render_consts.ship_width
        player_tops = player_bots + params.render_consts.ship_height

        enemy_bullet_lefts = (state.bullet_state.enemy_bullet_positions[:, params.x_idx, ...] * params.render_consts.screen_width) - (params.render_consts.bullet_width / 2)   # (2, 2, 5, 5)
        enemy_bullet_bots = (state.bullet_state.enemy_bullet_positions[:, params.y_idx, ...]  * params.render_consts.screen_height) - (params.render_consts.bullet_height / 2) 
        enemy_bullet_rights = enemy_bullet_lefts + params.render_consts.bullet_width
        enemy_bullet_tops = enemy_bullet_bots + params.render_consts.bullet_height

        no_overlap_x = (enemy_bullet_lefts >= player_rights) | (player_lefts >= enemy_bullet_rights)
        no_overlap_y = (enemy_bullet_bots >= player_tops) | (player_bots >= enemy_bullet_tops)
        
        enemy_bullet_player_intersections = jnp.logical_not(jnp.logical_or(no_overlap_x, no_overlap_y)) # (3, 2, 2, 5, 5)

        player_hit_by_any_enemy = jnp.any(enemy_bullet_player_intersections, axis=[1, 2, 3, 4]) # (3,)

        new_player_frames_since_death = jax.lax.select(
            player_hit_by_any_enemy, 
            jnp.zeros_like(state.history.player_frames_since_death),
            state.history.player_frames_since_death
        )

        new_players_active = jax.lax.select(
            player_hit_by_any_enemy,
            jnp.zeros_like(state.player_state.players_active),
            state.player_state.players_active
        )
        
        # If a plaer was hit, their position is inf. Else, select their current position. 
        player_hit_by_any_enemy_broadcasted = jnp.broadcast_to(
            player_hit_by_any_enemy[:, None],
            state.player_state.player_positions.shape
        )
        
        new_player_positions = jax.lax.select(
            player_hit_by_any_enemy_broadcasted,
            jnp.full_like(state.player_state.player_positions, jnp.inf),
            state.player_state.player_positions
        )
        
        # If an enemy was hit, it's frames since death is 0. Else, select its current frames since death value.
        enemy_bullet_hit_player = jnp.any(enemy_bullet_player_intersections, axis=0) # (2, 2, 5, 5)

        enemy_bullet_hit_player_broadcasted = jnp.broadcast_to(
            enemy_bullet_hit_player[:, None, ...], # [2, 1, 2, 5, 5] --> [2, 2, 2, 5, 5]
            state.bullet_state.enemy_bullet_positions.shape
        ) 

        # Enemy bullets that players are no longer active ...
        new_enemy_bullet_actives = jax.lax.select(
            enemy_bullet_hit_player, 
            jnp.zeros_like(state.bullet_state.enemy_bullet_actives), 
            state.bullet_state.enemy_bullet_actives
        ) 
        
        # and have their position set to inf. 
        new_enemy_bullet_positions = jax.lax.select(
            enemy_bullet_hit_player_broadcasted,
            jnp.full_like(state.bullet_state.enemy_bullet_positions, jnp.inf),
            state.bullet_state.enemy_bullet_positions
        )

        # Get the player hit reward (-100) if you've been hit by an enemy
        new_player_hit_rewards = params.reward_consts.player_hit_reward * player_hit_by_any_enemy

        return state.replace(
            player_state=state.player_state.replace(player_positions=new_player_positions,
                                                    players_active=new_players_active.astype(dtype=int)),
            history=state.history.replace(player_frames_since_death=new_player_frames_since_death),
            bullet_state=state.bullet_state.replace(enemy_bullet_actives=new_enemy_bullet_actives,
                                                    enemy_bullet_positions=new_enemy_bullet_positions),
            step_info=state.step_info.replace(player_hit_rewards=new_player_hit_rewards)
        )

    def _player_respawn_handler(self, state: EnvState, params: EnvParams):  
        # TODO: Initial values should be in params instead of state since they don't vary. 

        # If a player is active, keep its current frames since death (-1). Otherwise, add one to their frames since death value
        updated_frames_since_death = jax.lax.select(
            state.player_state.players_active == 1, 
            state.history.player_frames_since_death, 
            state.history.player_frames_since_death + 1
        )
        
        # Allow a player to respawn if their updated frames since death is greater than the respawn frequency threshold
        respawn_cond = jnp.greater_equal(
            updated_frames_since_death, 
            params.dynamics_consts.respawn_frequency
        )
        
        # If a player can respawn, their frames since death value is -1. Otherwise, keep the updated frames since death value. 
        new_player_frames_since_death = jax.lax.select(
            respawn_cond,
            jnp.full_like(respawn_cond, -1, dtype=int), 
            updated_frames_since_death
        )
        
        # Players are active if their frames since death is -1 
        new_players_active = jnp.equal(new_player_frames_since_death, -1).astype(dtype=int)

        # If player can respawn, respawn them at initial position. Else, keep them where they are currently.
        respawn_cond = respawn_cond[:, None] # [3, 1]
        respawn_cond = jnp.broadcast_to(
            respawn_cond, 
            shape=state.player_state.initial_player_positions.shape
        ) # [3, 2]

        new_player_positions = jax.lax.select(
            respawn_cond,
            state.player_state.initial_player_positions,
            state.player_state.player_positions
        )

        return state.replace(
            player_state=state.player_state.replace(
                players_active=new_players_active,
                player_positions=new_player_positions
            ),
            history=state.history.replace(
                player_frames_since_death=new_player_frames_since_death
            )
        )

    def _enemy_respawn_handler(self, state: EnvState, params: EnvParams):  
        
        # If an enemy is active, keep its current frames since death (-1). Otherwise, add one to their frames since death value
        updated_frames_since_death = jax.lax.select(
            state.enemy_state.enemies_active == 1, 
            state.history.enemy_frames_since_death,
            state.history.enemy_frames_since_death + 1
        )
        
        # Allow an enemy to respawn if their updated frames since death is greater than their respawn frequency threshold
        respawn_cond = jnp.greater_equal(
            updated_frames_since_death, 
            params.dynamics_consts.respawn_frequency
        )

        # If an enemy can respawn, their frames since death value is -1. Otherwise, keep the updated frames since death value.
        new_enemy_frames_since_death = jax.lax.select(
            respawn_cond, 
            jnp.full_like(respawn_cond, -1, dtype=int), 
            updated_frames_since_death
        )
        
        # Enemies are active if their frames since death is -1. 
        new_enemies_active = jnp.equal(new_enemy_frames_since_death, -1).astype(dtype=int)

        # If enemy just respawned, respawn them at initial position. Else, keep them where they are currently.
        respawn_cond = respawn_cond[None, ...] # [1, 2, 5, 5]
        respawn_cond = jnp.broadcast_to(
            respawn_cond, 
            shape=state.enemy_state.initial_enemy_positions.shape
        ) # [2, 2, 5, 5]

        new_enemy_positions = jax.lax.select(
            respawn_cond,
            state.enemy_state.initial_enemy_positions,
            state.enemy_state.enemy_positions
        )

        return state.replace(
            enemy_state=state.enemy_state.replace(enemies_active=new_enemies_active,
                                                  enemy_positions=new_enemy_positions),
            history=state.history.replace(enemy_frames_since_death=new_enemy_frames_since_death),
        )

    def _score_handler(self, state: EnvState, params: EnvParams):  # Updates based on score changes from previous handlers
         # TODO: There's def a better way to write this 
        # + 1 because frame hasn't been incremented yet
        is_terminal_step = state.time_state.frame_num + 1 >= params.dynamics_consts.max_num_frames_per_game
        # player_victory_rewards = state.player_victory_rewards 
        human_total_score = state.score_state.scores[params.human_idx] + state.score_state.scores[params.nao_for_human_score_idx]
        shutter_total_score = state.score_state.scores[params.shutter_idx] + state.score_state.scores[params.nao_for_shutter_score_idx]
        leading_player = jnp.array([jnp.less(human_total_score, shutter_total_score), jnp.greater(human_total_score, shutter_total_score), False], dtype=int).flatten()

        new_player_victory_rewards = jax.lax.cond(
            is_terminal_step,
            lambda _: leading_player * params.reward_consts.victory_reward,
            lambda _: jnp.array([0, 0, 0]),
            None
        )

        return state.replace(
            step_info=state.step_info.replace(player_victory_rewards=new_player_victory_rewards),
        )
    
    def _reward_handler(self, state: EnvState, params: EnvParams):
        enemy_elimination_reward = state.step_info.enemy_elimination_rewards
        player_hit_reward = state.step_info.player_hit_rewards
        player_taking_score_lead_reward = state.step_info.player_taking_score_lead_rewards
        player_victory_reward = state.step_info.player_victory_rewards
        player_rewards = enemy_elimination_reward + player_hit_reward + player_taking_score_lead_reward + player_victory_reward
        return player_rewards
        

    def _update_time(self, state: EnvState, params: EnvParams):
        new_frame_num = state.time_state.frame_num + 1
        return state.replace(
            time_state = state.time_state.replace(frame_num = new_frame_num)
        )
    
    def _to_observation(self, state: EnvState, params: EnvParams):
        pass

    def _is_terminal(self, state: EnvState, params: EnvParams):
        # state.time_state.frame_num + 1 >= params.dynamics_consts.max_num_frames_per_game
        False
    
    def _get_info(self, state: EnvState, params: EnvParams):
        pass

    # TODO: Finish rendering functions
    #       probably fine just to keep screen in state func... blitting isn't major time bottle neck I think
    #       main issue is idk if 
    def render_scores(self):
        pass

    def render(self):
        pass
