import random
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from agents.nao_policy import NaoPolicy

#Changes made agent = 'human' and reward model from full to scores because of parallelization. Fixed boxes on bullets as they were too small, fixed respawnign that happened when non human agents were hit. Added new reward function
class CooperativeSpaceInvadersEnv(gym.Env):
    metadata = {
        "name": "SpaceInvaders_v0",
        "render_modes": ["rgb_array","human"], 
        "render_fps": 4
    }

    def __init__(self, agent='Human' , verbose = False, reward_model='full',render_mode ="rgb_array", support_policy = 'equal'):
        """
        Arguments:
            agent -> which agent is being trained
            verbose -> whether or not information is to be returned 
            reward_model -> what reward model the agent is going to train 
        """
        super(CooperativeSpaceInvadersEnv, self).__init__()

        #argument handling 
        self.verbose = verbose
        self.reward_model = reward_model
        self.render_mode = render_mode
        self.support_policy = support_policy


        #rendering objects
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        try:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            print("Successfully setup pygame rendering.")
        except pygame.error as e:
            print(f"Error setting up pygame rendering: {e}")
            print("Skipping rendering.")        
        pygame.display.set_caption("Custom Environment")
        self.color_bullet = (0, 255, 0)    # Green
        #Use Images from Multiplayer SI Game
        self.ship_image = pygame.image.load('assets/images/ship.png')
        self.ship_image = pygame.transform.scale(self.ship_image, (50, 48))  
        self.shutter_image = pygame.image.load('assets/images/ship2.png')
        self.shutter_image = pygame.transform.scale(self.shutter_image, (50, 48))  
        self.nao_image = pygame.image.load('assets/images/shield.png')
        self.nao_image = pygame.transform.scale(self.nao_image, (50, 48)) 
        self.enemy_image = pygame.image.load('assets/images/enemy1_1.png')
        self.enemy_image = pygame.transform.scale(self.enemy_image, (50, 48)) 
        self.enemy_image2 = pygame.image.load('assets/images/enemy3_1.png')
        self.enemy_image2 = pygame.transform.scale(self.enemy_image2, (50, 48)) 
        self.enemy_image3 = pygame.image.load('assets/images/coop_enemy_angry.png')
        self.enemy_image3 = pygame.transform.scale(self.enemy_image3, (50, 48)) 
        self.enemy_image4 = pygame.image.load('assets/images/coop_enemy_calm.png')
        self.enemy_image4 = pygame.transform.scale(self.enemy_image4, (50, 48)) 
        self.bullet_image = pygame.image.load('assets/images/laser.png')
        self.bullet_image = pygame.transform.scale(self.bullet_image, (5, 15))  
        self.enemy_bullet_image = pygame.image.load('assets/images/enemylaser.png')
        self.enemy_bullet_image = pygame.transform.scale(self.enemy_bullet_image, (5, 15)) 
        #Score and their fonts, we are using a custom font
        self.font = pygame.font.Font("assets/fonts/custom/joystix.ttf", 25)  # You can replace None with 'path/to/your/font.ttf'

        # Observations are dictionaries 
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(111,), dtype=np.float32)

        # We have 4 actions, corresponding to 'left', 'right', 'shoot' and 'nothing'
        self.action_space = spaces.Discrete(4)

        # Define initialisation parameters
        self.initial_human_position_x = 0.25 
        self.initial_human_position_y = 0.9 
        self.initial_shutter_position_x = 0.868 
        self.initial_shutter_position_y = 0.9
        self.initial_nao_position_x = 0.5 #check
        self.initial_nao_position_y = 0.9
        self.enemies_x = [0.0314, 0.1062,0.1814,0.2564,0.331,0.0314, 0.1062,0.1814,0.2564,0.331,0.0314, 0.1062,0.1814,0.2564,0.331,0.0314, 0.1062,0.1814,0.2564,0.331,0.0314, 0.1062,0.1814,0.2564,0.331,0.581,0.655,0.7314,0.8064,0.881,0.581,0.655,0.7314,0.8064,0.881,0.581,0.655,0.7314,0.8064,0.881,0.581,0.655,0.7314,0.8064,0.881,0.581,0.655,0.7314,0.8064,0.881]
        self.human_enemies_x = [0.0314, 0.1062,0.1814,0.2564,0.331] #Enemies on human side only
        self.enemies_y = [0.543,0.543,0.543,0.543,0.543,0.460,0.460,0.460,0.460,0.460,0.376,0.376,0.376,0.376,0.376,0.293,0.293,0.293,0.293,0.293,0.210,0.210,0.210,0.210,0.210,0.543,0.543,0.543,0.543,0.543,0.460,0.460,0.460,0.460,0.460,0.376,0.376,0.376,0.376,0.376,0.293,0.293,0.293,0.293,0.293,0.210,0.210,0.210,0.210,0.210]
        self.active_enemies_left_side = []
        self.active_enemies_right_side = []
        self.shutter_shoots = True
        self.nao_shoots = True

        # Define state variables
        self.human_position_x = self.initial_human_position_x
        self.human_position_y = self.initial_human_position_y
        self.shutter_position_x = self.initial_shutter_position_x
        self.shutter_position_y = self.initial_shutter_position_y
        self.nao_position_x = self.initial_nao_position_x
        self.nao_position_y = self.initial_nao_position_y
        self.enemies_active = [True] * len(self.enemies_x)
        self.bullets_x = None
        self.bullets_y = None
        self.enemy_bullets_x = []
        self.enemy_bullets_y = []
        self.scores = {'Human': 0, 'Shutter': 0, 'Nao': 0}
        self.single_turn_score = {'Human': 0, 'Shutter': 0, 'Nao': 0}

        self.time = None
        self.frame = 0

        #Define Threshold variables
        self.VERTICAL_BUFFER  = 440 / 600 #600 is screen height 800  is width
        self.HIT_RANGE  =  35/800 # range for dodging bullets- normalized
        self.SECOND_HIT_RANGE = 50/800 #normalized
        self.min_x = 0
        self.SHOOTING_RANGE = 24/800 #normalized
        self.NAO_RELATIVE_SPEED = 0.7 # relatively how much shorter time between bullets is for Shutter and Nao
        self.SHUTTER_RELATIVE_SPEED = .8
        self.FREQUENCY_BOUND = 3000 # max time between shots
        self.max_human_shoot_frequency = 400
        self.bullet_velocity = 0.00277
        self.enemy_height = 48 / self.screen_height
        self.enemy_width = 50 / self.screen_width
        self.spaceship_left_edge= self.human_position_x - self.enemy_width / 2 #human ship left edge
        self.spaceship_right_edge = self.human_position_x + self.enemy_width / 2 #human ship ridht edge
        self.spaceship_upper_edge = (self.human_position_y + self.enemy_height/2) #Human ship +adding 1/2 enemy height


        
        # Define variables to assist with shooting and bullet states
        self.enemies_hit_frames = [-1] * len(self.enemies_x) 
        self.shoot_frequency = 40#50
        self.shoot_counter = 0
        self.bullets_left_side = []
        self.bullets_right_side = []
        self.average_shot_frequency = {'Human': None, 'Shutter': None, 'Nao': None}  # Initialize average frequencies
        self.last_shot_time = {'Human': 0, 'Shutter': 0, 'Nao': 0}  
        self.shot_time_history = {'Human': [], 'Shutter': [], 'Nao': []}  # Initialize shot time history for each agent
        #self.player_avg_frequency = self.FREQUENCY_BOUND
        self.enemy_bullets = np.array([[0.,0.]] * 10)
        self.enemyb_free = list(range(10))
        self.enemyb_used = []
        self.player_bullets = np.array([[0.,0.,None]]*10)
        self.human_distance_to_bullets = [0] *10
        self.small_distance_to_bullets = [0]*3
        self.nao_increasing_player_score = [0] *2
        self.playerb_free = list(range(10))
        self.playerb_used = []
        self.bullet_y_positions = np.zeros((10, 2))  # 20 columns and 3 slots per column
        self.bullet_threat_binary = [0] *5
        self.time_until_collision =[float(1000)] * 5
        

        # Agent handling
        self.agents = ['Human', 'Shutter', 'Nao'] # List of possible agent
        assert agent in self.agents, "Agents should be selected from ['Human', 'Shutter', 'Nao']" # Check that given agent is valid
        self.agent_selected = agent  
        self.agents_unselected = [ag for ag in self.agents if ag != self.agent_selected] # Define unselected agents
        self.active = {'Human': True, 'Shutter': True, 'Nao': True}
        self.elimination_reward = {'Human': 0, 'Shutter': 0, 'Nao': 0}
        self.support = 1 #Nao Support

        #Reward handling
        self.rewards = {agent: 0. for agent in self.agents}
        #self.bullet_y_positions = np.zeros(20) 

        #Time Variables
        self.start_time = time.time()
        self.clock = pygame.time.Clock() 
        self.last_enemy_shot_time = time.time()
        self.running_avg_reward = 0
        self.sum_reward =0
        self.human_hit_count = 0
        
        # Define shooting rates for two groups
        self.fast_shooting_rate = 0.05  # Faster shooting rate
        self.slow_shooting_rate = 0.3  # Slower shooting rate

        # Define column groups
        self.group1_indexes = [25, 84, 145, 205, 264]  
        self.group2_indexes = [464, 524, 585, 645, 704]

        # Create a mapping from pixel position to column index
        self.col_index_map = {col: idx for idx, col in enumerate(self.group1_indexes + self.group2_indexes)}

        # Flag to determine which group shoots faster
        self.group1_fast = True  # Initial state
        self.frame_counter = 0   # Frame counter

        #Nao Policy
        self.nao_policy_instance = NaoPolicy(
            self.initial_nao_position_y, self.screen_width, self.screen_height, self.VERTICAL_BUFFER,
            self.HIT_RANGE, self.SECOND_HIT_RANGE, self.SHOOTING_RANGE, self.NAO_RELATIVE_SPEED, self.FREQUENCY_BOUND,self.support_policy
        )


    def reset(self, seed=None, options=None):
        """
        Reset game to default values 
        """
        # Reset all variables

        
        self.time = 0
        self.frame = 0
        self.human_position_x = self.initial_human_position_x
        self.human_position_y = self.initial_human_position_y
        self.shutter_position_x = self.initial_shutter_position_x
        self.shutter_position_y = self.initial_shutter_position_y
        self.nao_position_x = self.initial_nao_position_x
        self.nao_position_y = self.initial_nao_position_y
        self.active = {'Human': True, 'Shutter': True, 'Nao': True}
        self.scores = {'Human': 0, 'Shutter': 0, 'Nao': 0}
        self.single_turn_score = {'Human': 0, 'Shutter': 0, 'Nao': 0}
        self.enemies_hit_frames = [-1] * len(self.enemies_x)
        self.bullets_x = []
        self.bullets_y = []
        self.enemies_active = [True] * len(self.enemies_x)
        self.enemy_bullets_x = []
        self.enemy_bullets_y = []
        self.enemy_bullets_x_rel = self.enemy_bullets_x
        self.enemy_bullets_y_rel = self.enemy_bullets_y
        self.elimination_reward = {'Human': 0, 'Shutter': 0, 'Nao': 0}
        self.bullets_left_side = []
        self.bullets_right_side = []
        self.shutter_shoots = True
        self.nao_shoots = True

        observation = self._get_obs()

        info = self._get_info() if self.verbose == True else {}
        current_time_ms = time.time() * 1000
        self.last_shot_time = {agent: current_time_ms for agent in self.agents}  # Initialize with current time
        self.enemy_bullets = np.array([[0.,0.]] * 10)
        self.enemyb_free = list(range(10))
        self.enemyb_used = []
        self.player_bullets = np.array([[0.,0.,None]]*10)
        self.playerb_free = list(range(10))
        self.playerb_used = []
        self.rewards = {agent: 0. for agent in self.agents}
        self.last_enemy_shot_time = time.time()
        #self.bullet_y_positions = np.zeros(20) 
        self.bullet_y_positions = np.zeros((10, 2))  # 20 columns and 3 slots per column- changed it
        self.human_distance_to_bullets = [0] *10
        self.small_distance_to_bullets = [0]*3


        #Reset counts
        self.running_avg_reward = 0
        self.sum_reward  = 0
        self.human_hit_count = 0
        self.bullet_threat_binary = [0] *5
        self.time_until_collision =[float(1000)] * 5
        self.nao_increasing_player_score = [0] *2


        return observation, info

    def step(self, action):
        """
        Perform a step in the game 
        """
        #self.agent_selected = 'Human'
        #self.clock.tick(60)  # This will ensure that step() doesn't execute more than 60 times per second. without this the game is much faster and 124 fps on my mac. This adjusts it to the original game.
        #Implement actions for all agents.
        #     # Increment frame counter
        self.frame_counter += 1

        # Change shooting rate flag every 100 frames
        if self.frame_counter % 1000 == 0:
            
            self.group1_fast = not self.group1_fast
            if self.group1_fast:
                group1_rate = self.fast_shooting_rate
                group2_rate = self.slow_shooting_rate
                #print(f"Group 1 shooting rate: {group1_rate}, Group 2 shooting rate: {group2_rate}")

            else:
                group1_rate = self.slow_shooting_rate
                group2_rate = self.fast_shooting_rate
                #print(f"Group 1 shooting rate: {group1_rate}, Group 2 shooting rate: {group2_rate}")


        left, right, shoot, nearest_enemy_nao = self.nao_policy_instance.nao_policy(
            self.frame, self.nao_position_x, self.bullets_left_side, self.bullets_right_side,
            self.enemies_active, self.enemies_x, self.enemies_y, self.average_shot_frequency, self.last_shot_time
        )
        #left,right,shoot, nearest_enemy_nao= self.nao_policy()
        left_shutter, right_shutter, shoot_shutter = self.shutter_policy(nearest_enemy_nao)
        self.action_handler(action,left,right,shoot,left_shutter, right_shutter, shoot_shutter) 
        #Adjust the state at every step
        self.enemy_shot_handler() # Implement enemy shooting actions
        self.update_bullet_positions() # Update the position of bullets 
        self.check_collisions() # Check if any bullets have collided with enemies or players
        self.human_respawn_handler() # Check for destroyed enemies that need to respawn
        self.enemy_respawn_handler() # Check for destroyed enemies that need to respawn
        self.reward_handler() # Calculate rewards for the frame
        self.frame += 1 # Update frame count
        if self.frame %60 ==0: #increment a second every 60 frames
            self.time +=1
        observation = self._get_obs() # Get observation 

        #Get Reward
        #print('agent selected',self.agent_selected)
        reward = self.rewards[self.agent_selected] #reward


        #Terminate if 7,200 frames (2 minutes) have passed 
        terminated = self.frame > 7200 #10000 Terminate after 7,200 frames (or 2 mins)

        #Add information   about the state
        self.sum_reward += reward
        info = self._get_info() if self.verbose == True else {}
        self.sum_reward += reward
        self.running_avg_reward = self.sum_reward/self.frame
        if self.frame ==7200: #Optional: to see how well it is performing as it is training
            print('Episode Rewards:', self.sum_reward)
            print('average reward:',self.running_avg_reward )
            print('number of hits:' ,self.human_hit_count)

        if self.enemyb_free == []:
            print('WARNING')

        # Reset rewards for next frame
        self.rewards[self.agent_selected] =0

        return observation, reward, terminated, False, info 
    
    def shutter_policy(self,nearest_enemy_nao):
        """
        Policy for Shutter. Identifies nearest bullets and enemies and adjusts its position or shoots.
        """
        #initialize variables for actions
        left = False
        right = False
        shoot = False
        hit = False
        nearest_bullet = [0,0]

        #This variable lets us know whether shutter ship can shoot or not
        self.shutter_shoots = True
        current_time_ms = time.time() * 1000
        shutter_last_shot = 0


        #Get the average shooting frequency of human ship
        player_avg_frequency = self.FREQUENCY_BOUND
        if self.average_shot_frequency['Human']:
            player_avg_frequency = min(self.FREQUENCY_BOUND, int(self.average_shot_frequency['Human']))
        if self.shot_time_history['Shutter']:
            shutter_last_shot = self.shot_time_history['Shutter'][-1]

        x_diff_prev= 1 #self.screen_width/self.screen_width 

        #Find nearest bullet only in the right side of the screen
        for bullet in self.bullets_right_side:
            x_diff = abs(bullet[0]-self.shutter_position_x)
            if bullet[1]< self.initial_shutter_position_y and bullet[1]> self.VERTICAL_BUFFER and x_diff<self.HIT_RANGE *2:
                if x_diff < x_diff_prev:
                    nearest_bullet= bullet
                    x_diff_prev = x_diff


        #look through active enemies and assign them based on whether they are on the left or right side
        self.active_enemies_right_side.clear()
        self.active_enemies_left_side.clear()
        for (x, y), active in zip(zip(self.enemies_x, self.enemies_y), self.enemies_active):
            if active:
                if x >= 0.5:
                    self.active_enemies_right_side.append((x,y))
                else:
                    self.active_enemies_left_side.append((x,y))

        # Find nearest enemy by Y
        nearest_enemy = [0,0]
        nearest_x_diff = 1 #self.screen_width
        nearest_y_diff = self.VERTICAL_BUFFER
        closest_x_diff =1 # self.screen_width

        #In the original version they looked through each enemy and checked if it was active or not. We are just going to check active enmies. 
        #Find nearest enemy for Shutter Ship
        for enemy in self.active_enemies_right_side:
            if enemy == nearest_enemy_nao:
                pass
            else:
                check_distance_x = abs(enemy[0] - self.shutter_position_x)
                check_distance_y = abs(enemy[1] - self.initial_shutter_position_y)

                #index = self.active_enemies_right_side.index(enemy)
                if check_distance_y < nearest_y_diff:
                    nearest_enemy = enemy
                    nearest_x_diff = check_distance_x
                    nearest_y_diff = check_distance_y
                elif check_distance_y == nearest_y_diff and check_distance_x < nearest_x_diff:
                    nearest_enemy = enemy
                    nearest_x_diff = check_distance_x
                    nearest_y_diff = check_distance_y
                elif check_distance_y < (nearest_y_diff+(50/self.screen_height)) and check_distance_x < closest_x_diff :
                    closest_x_diff = check_distance_x

        if round(time.time() * 1000) - self.last_shot_time['Shutter'] < player_avg_frequency * self.SHUTTER_RELATIVE_SPEED:
            self.shutter_shoots = False 

        # check if agent is in danger of being hit by bullet
        if nearest_bullet[0] <= self.shutter_position_x + self.HIT_RANGE and nearest_bullet[0] >= self.shutter_position_x - self.HIT_RANGE:
            hit = True

        approached_enemy = False

        if not(self.shutter_shoots):
            #If agent cannot shoot, adjust position
            if hit:
                if nearest_bullet[0] >= 1 - (75/800):
                    left = True
                elif nearest_bullet[0] <= self.min_x + (55/800):
                    right = True
                elif nearest_bullet[0] > self.shutter_position_x:
                    left = True
                elif nearest_bullet[0] <= self.shutter_position_x:
                    right = True
        else:
            #IF agent CAN SHOOT
            if nearest_x_diff <= self.SHOOTING_RANGE or closest_x_diff <= self.SHOOTING_RANGE:
                shoot = True

            if nearest_enemy[0] < self.shutter_position_x:
                if not(nearest_bullet[0] < self.shutter_position_x and self.initial_shutter_position_y - nearest_bullet[1] < (200/300) and self.shutter_position_x - nearest_bullet[0] <= self.SECOND_HIT_RANGE):
                    left = True
                    approached_enemy = True
            elif nearest_enemy[0] > self.shutter_position_x:
                if not(nearest_bullet[0] > self.shutter_position_x and self.initial_shutter_position_y - nearest_bullet[1] < (200/300) and nearest_bullet[0] - self.shutter_position_x <= self.SECOND_HIT_RANGE):
                    right = True
                    approached_enemy = True
                    
            if not approached_enemy and hit:
                if x_diff_prev > (5/self.screen_width):
                    # if not going to hit bullet, don't shoot so you can move
                    shoot = False
                # figure out which way to move so you don't get hit  
                if nearest_bullet[0] >= 1-(75/self.screen_width):
                    left = True
                elif nearest_bullet[0] <= self.min_x + (55/self.screen_width):
                    right = True
                elif nearest_bullet[0] > self.shutter_position_x:
                    left = True
                elif nearest_bullet[0] <= self.shutter_position_x:
                    right = True

        return left,right, shoot

        
    
    def nao_policy(self):
        """
        Policy for Nao. Identifies nearest bullets and enemies and adjusts its position or shoots.
        """
        #initialize variables for actions
        left = False
        right = False
        shoot = False
        self.nao_shoots = True
        hit = False
        nearest_bullet = [0,0]
        current_time_ms = time.time() * 1000
        nao_last_shot = 0

        # Toggle support every 1000 frames. Nao switches support every 1000 frames
        if self.frame % 1000 == 0:
            self.support = 1 if self.support == 0 else 0

        #Get the average shooting frequency of human ship
        player_avg_frequency = self.FREQUENCY_BOUND
        if self.average_shot_frequency['Human']:
            player_avg_frequency = min(self.FREQUENCY_BOUND, int(self.average_shot_frequency['Human']))

        if self.shot_time_history['Nao']:
            nao_last_shot = self.shot_time_history['Nao'][-1]

        if self.support == 1:
            #Limit search to left side bullets
            bullets_to_search = self.bullets_left_side
        else:
            #Limit search to right side bullets
            bullets_to_search = self.bullets_right_side


        #Find Nearest bullet for Nao Ship
        x_diff_prev= 1 #self.screen_width/self.screen_width
        for bullet in bullets_to_search:
            x_diff = abs(bullet[0]-self.nao_position_x)
            if bullet[1]< self.initial_nao_position_y and bullet[1]> self.VERTICAL_BUFFER and x_diff<self.HIT_RANGE *2:
                if x_diff < x_diff_prev:
                    nearest_bullet= bullet
                    x_diff_prev = x_diff

        #look through active enemies and assign them to the left and right side
        self.active_enemies_right_side.clear()
        self.active_enemies_left_side.clear()
        for (x, y), active in zip(zip(self.enemies_x, self.enemies_y), self.enemies_active):
            if active:
                if x >= 0.5:
                    self.active_enemies_right_side.append((x,y))
                else:
                    self.active_enemies_left_side.append((x,y))

        if self.support == 1:
            #only search through enemies on left side
            enemies_to_search = self.active_enemies_left_side
        else:
            #only search through enemies on right side
            enemies_to_search = self.active_enemies_right_side

        # # find nearest enemy by Y
        nearest_enemy = [0,0]
        nearest_x_diff = 1 #self.screen_width
        nearest_y_diff = self.VERTICAL_BUFFER
        closest_x_diff =1 # self.screen_width


        #In the Js version they looked through each enemy and checked if it was active or not. We are just going to check active enmies. 
        #Find nearest enemy to Nao Ship
        for enemy in enemies_to_search:
            check_distance_x = abs(enemy[0] - self.nao_position_x)
            check_distance_y = abs(enemy[1] - self.initial_nao_position_y)
            if check_distance_y < nearest_y_diff:
                nearest_enemy = enemy
                nearest_x_diff = check_distance_x
                nearest_y_diff = check_distance_y
            elif check_distance_y == nearest_y_diff and check_distance_x < nearest_x_diff:
                nearest_enemy = enemy
                nearest_x_diff = check_distance_x
                nearest_y_diff = check_distance_y
            elif check_distance_y < (nearest_y_diff+(50/self.screen_height)) and check_distance_x < closest_x_diff :
                closest_x_diff = check_distance_x

        nearest_enemy_nao = nearest_enemy

        if round(time.time() * 1000) - self.last_shot_time['Nao'] < player_avg_frequency * self.NAO_RELATIVE_SPEED:
            self.nao_shoots = False 

        # check if ai_agent is in danger of being hit by bullet
        if nearest_bullet[0] <= self.nao_position_x + self.HIT_RANGE and nearest_bullet[0] >= self.nao_position_x - self.HIT_RANGE:
            hit = True

        approached_enemy = False


        if not(self.nao_shoots):            
            # if can't shoot, figure out which way to move; nearest_bullet[0] >= self.screen_width - (75/800)
            if hit:
                if nearest_bullet[0] >= 1 - (75/800):
                    left = True
                elif nearest_bullet[0] <= self.min_x + (55/800):
                    right = True
                elif nearest_bullet[0] > self.nao_position_x:
                    left = True
                elif nearest_bullet[0] <= self.nao_position_x:
                    right = True
        else:
            #IF AI CAN SHOOT
            if nearest_x_diff <= self.SHOOTING_RANGE or closest_x_diff <= self.SHOOTING_RANGE:
                shoot = True

            if nearest_enemy[0] < self.nao_position_x:
                if not(nearest_bullet[0] < self.nao_position_x and self.initial_nao_position_y - nearest_bullet[1] < (200/self.screen_height) and self.nao_position_x - nearest_bullet[0] <= self.SECOND_HIT_RANGE):
                    left = True
                    approached_enemy = True
            elif nearest_enemy[0] > self.nao_position_x:
                if not(nearest_bullet[0] > self.nao_position_x and self.initial_nao_position_y - nearest_bullet[1] < (200/self.screen_height) and nearest_bullet[0] - self.nao_position_x <= self.SECOND_HIT_RANGE):
                    right = True
                    approached_enemy = True
                    
            if not approached_enemy and hit:
                if x_diff_prev > (5/self.screen_width):
                    # if not going to hit bullet, don't shoot so you can move
                    shoot = False
                # figure out which way to move so you don't get hit
                if nearest_bullet[0] >= 1-(75/self.screen_width):
                    left = True
                elif nearest_bullet[0] <= self.min_x + (55/self.screen_width):
                    right = True
                elif nearest_bullet[0] > self.nao_position_x:
                    left = True
                elif nearest_bullet[0] <= self.nao_position_x:
                    right = True
            
        return left,right, shoot, nearest_enemy_nao
    

    def bullet_threat_check(self):
        """"
        Checks whether the human spaceship is aligned in the same column as an enemy bullet. Also checks how much time before the bullet collides with the human spaceship
        """
        #ensure that these are set to zero or a large number
        self.bullet_threat_binary = [0] * 5
        self.time_until_collision =[float(1000)] * 5


        for col in range(5):
            bullet_x = self.human_enemies_x[col]  # Get the x position of the enemy in this column
            # Check both bullets in the column
            for bullet_index in range(2):
                bullet_y = self.bullet_y_positions[col, bullet_index]

                # Check if there is an active bullet (non-zero y position)
                if bullet_y > 0:
                    # Since the y-position of the spaceship is fixed, we only need to check the x-position
                    if self.spaceship_left_edge <= bullet_x <= self.spaceship_right_edge:
                        distance_to_travel = self.spaceship_upper_edge- bullet_y
                        if distance_to_travel > 0:
                            # Calculate the time until collision
                            time_to_collision = distance_to_travel / self.bullet_velocity
                            # Update the time until collision for the column if this bullet is a closer threat
                            if time_to_collision < self.time_until_collision[col]:
                                self.time_until_collision[col] = time_to_collision/262
                                # Mark this column as a threat if any bullet in this column can hit the spaceship
                                self.bullet_threat_binary[col] = 1

        self.time_until_collision = [0 if value > 1 else value for value in self.time_until_collision]

    
    def _get_obs(self):
        """
        Capture the state of the game.
        """

        #Position of the human ship: the leftmost, and rightmost, edge of the human sprite
        observation = [self.human_position_x]
        observation += [self.human_position_x - (self.enemy_width / 2)]  # Left edge
        observation += [self.human_position_x + (self.enemy_width / 2)]  # Right edge
        #Shutter's position
        observation += [self.shutter_position_x]
        #Nao's Position
        observation += [self.nao_position_x]
        #Enemy Positions
        observation += [float(val) for val in self.enemies_active]

        #Player bullets
        observation += list(self.player_bullets[:,:2].flatten())
        #Enemy bullets
        flattened_enemy_bullets = self.bullet_y_positions.flatten(order='F')#Enemy bullets sorted by column
        observation += list(flattened_enemy_bullets)
        #Human distance to bullets
        observation += self.small_distance_to_bullets

        #Time in the game
        observation += [(self.time/120)]

        #Check for bullet threats to the human spaceship and update the bullet_threat_binary and time_until collision
        self.bullet_threat_check()
        observation += list(self.time_until_collision)
        observation += list(self.bullet_threat_binary)
        
        #scores- normalized
        observation += list([value/3500 for key, value in list(self.scores.items())[:2]])

        #reset count
        self.nao_increasing_player_score = [0]*2

        return np.array(observation)
    

    
    def _get_info(self):
        """
        Return structured dictionary state information for troubleshooting
        """
        info = {
            'player_positions': {
                'Human': [self.human_position_x, self.human_position_y],
                'Shutter': [self.shutter_position_x, self.shutter_position_y],
                'Nao': [self.nao_position_x, self.nao_position_y],
            },
            'enemy_active': self.enemies_active,
            'bullet_positions': self.player_bullets, #check
            'enemy_bullet_positions': self.enemy_bullets, #check
            'scores': self.scores,
            'time': self.time,
            }
        
        return info

    def action_handler(self, action,left,right,shoot,left_shutter, right_shutter, shoot_shutter):
        """
        Handle all agent actions
        """
        # Human Agent
        if action == 0:
            self.move_left(self.agent_selected)
            
        elif action == 1:  #  '1' corresponds to 'move_right'
            self.move_right(self.agent_selected)
            
        elif action == 2:  # '2' corresponds to 'shoot'
            self.shoot(self.agent_selected)
            
        # For Shutter and Nao
        for agent in self.agents_unselected:
            if agent == "Shutter":
                #Check that only one action is taken
                if left_shutter:  
                    self.move_left(agent)

                if right_shutter:
                    self.move_right(agent)

                if shoot_shutter:
                    self.shoot(agent)

            else:
                #Nao actions
                if left:
                    self.move_left(agent)
                  
                if right:                  
                    self.move_right(agent)
   
                if shoot:           
                    self.shoot(agent)

    def move_left(self, agent):
        """
        Complete move left action
        """
        if agent == 'Human' and self.active['Human']:
            self.human_position_x = np.clip(self.human_position_x - 0.00625, 0.0002, 1) 
        elif agent == 'Shutter' and self.active['Shutter']:
            self.shutter_position_x = np.clip(self.shutter_position_x - 0.00625, .5, 1)
        elif agent == 'Nao' and self.active['Nao']:
            self.nao_position_x = np.clip(self.nao_position_x - 0.00625, 0, 1)

    def move_right(self, agent):
        """
        Complete move right action
        """
        if agent == 'Human' and self.active['Human']:
            self.human_position_x = np.clip(self.human_position_x + 0.00625, 0, .35) #0,.4 ##LIMITING HUMAN MOVEMENT TO ONE HALF OF THE SCREEN
        elif agent == 'Shutter' and self.active['Shutter']:
            self.shutter_position_x = np.clip(self.shutter_position_x + 0.00625, 0, 1)
        elif agent == 'Nao' and self.active['Nao']:
            self.nao_position_x = np.clip(self.nao_position_x + 0.00625, 0, 1)

    def shoot(self, agent): 
        """
        Complete shoot aciton
        """
        current_time = time.time() * 1000  # Current time in milliseconds
        time_since_last_shot = current_time - self.last_shot_time[agent]

        # Throttling shots based on agent type
        if agent == 'Human' and time_since_last_shot <=self.max_human_shoot_frequency:
            return
        
        if agent == 'Shutter' and not self.shutter_shoots:
            return
        
        if agent == 'Nao' and not self.nao_shoots:
            return

        # bullets move upwards at a fixed rate
        bullet_speed = -0.01
        #Create slots for bullets
        agent_slots = {
        'Human': [0, 1,2,3],
        'Shutter': [4, 5, 6],
        'Nao': [7, 8, 9]
        }
        # Find the lowest available index within the agent's designated slots
        available_slots = [idx for idx in agent_slots[agent] if idx in self.playerb_free]

        if self.playerb_free:
            if not available_slots:
                #print("No available slots for", agent)
                pass

            if available_slots:
                index =  min(available_slots)  # Get the lowest available slot
                self.player_bullets[index] = [self.get_agent_position(agent)[0], self.get_agent_position(agent)[1] + bullet_speed, agent]
                self.playerb_used.append(index)
                self.last_shot_time[agent] = current_time  # Correctly updating here
                self.playerb_free.remove(index)
            #Only track the last five bullets
            self.update_shot_time_history(agent, time_since_last_shot)
        else:
            print("No Free Bullet Slots")


    def update_shot_time_history(self, agent, time_since_last_shot):
        """
        Ensure the bullet history doesn't exceed 5 entries. Only tracking the last five bullets
        """ 
        if len(self.shot_time_history[agent]) >= 5:
            self.shot_time_history[agent].pop(0)
        self.shot_time_history[agent].append(time_since_last_shot)
        self.average_shot_frequency[agent] = sum(self.shot_time_history[agent]) / len(self.shot_time_history[agent])


    def get_agent_position(self, agent):
        """
        Given the name of an agent, return the position of said agent.
        """ 
        if agent == 'Human':
            return (self.human_position_x, self.human_position_y)
        if agent == 'Nao':
            return (self.nao_position_x, self.nao_position_y)
        if agent == 'Shutter':
            return (self.shutter_position_x, self.shutter_position_y)


    def reward_handler(self):
        """
        Calculate rewards for each agent.
        Add new reward models for reward tuning experiments.
        """
        if self.reward_model == 'full':
            # Iterate over enemies
            for agent in self.agents:
                if self.active[agent]:
                    # Add rewards for hitting an enemy
                    self.rewards[agent] += self.elimination_reward[agent]


                self.elimination_reward[agent] = 0

        else: 
            #factoring score increase
            for agent in self.agents:
                if self.active[agent]:
                    # Add rewards for hitting an enemy
                    if agent == 'Human':
                        self.rewards[agent] += (1*self.single_turn_score['Human']) + (-1 * self.single_turn_score['Shutter']) +self.elimination_reward['Human']
                self.elimination_reward[agent] = 0


    def enemy_shot_handler(self):
        """
        Handle enemy shots
        """
        self.shoot_counter += 1
        if self.shoot_counter >= self.shoot_frequency:  # Enemy can't shoot unless the shoot counter is greater than shoot frequency
            self.enemy_shoot()
            self.shoot_counter = 0  # Reset the counter after shooting

    def enemy_shoot(self):
        bullet_speed = 0.00277
        current_time = time.time()

        # Determine shooting rates based on the flag
        if self.group1_fast:
            group1_delay = self.fast_shooting_rate
            group2_delay = self.slow_shooting_rate
        else:
            group1_delay = self.slow_shooting_rate
            group2_delay = self.fast_shooting_rate


        # Find active enemies in both groups
        active_columns = {}
        for i, (x, y, active) in enumerate(zip(self.enemies_x, self.enemies_y, self.enemies_active)):
            if active:
                col_pixel = int(x * self.screen_width)
                if col_pixel in self.col_index_map:
                    # Keep track of the lowest active enemy in each column
                    if col_pixel not in active_columns or y > active_columns[col_pixel][1]:
                        active_columns[col_pixel] = (i, y)

        if not active_columns:
            print("No active columns with enemies to shoot from.")
            return

        # Randomly select an active enemy from the combined set
        chosen_col_pixel = random.choice(list(active_columns.keys()))
        lowest_enemy_idx, lowest_enemy_y = active_columns[chosen_col_pixel]

        # Determine the shooting rate based on the chosen enemy's group
        if chosen_col_pixel in self.group1_indexes:
            group_shooting_rate = group1_delay
        else:
            group_shooting_rate = group2_delay

        # If the delay hasn't passed, return
        if (current_time - self.last_enemy_shot_time) < group_shooting_rate:
            return

        if not self.enemyb_free:
            print("No bullets available to shoot.")
            return

        if self.enemyb_free:
            bullet_idx = self.enemyb_free.pop(0)
            enemy_x = self.enemies_x[lowest_enemy_idx]
            enemy_y = self.enemies_y[lowest_enemy_idx] + bullet_speed
            self.last_enemy_shot_time = time.time()
            self.enemy_bullets[bullet_idx] = [enemy_x, enemy_y]
            self.enemyb_used.append(bullet_idx)

            # Calculate index for bullet_y_positions
            col_idx = self.col_index_map[chosen_col_pixel]
            offset = 4 if col_idx < len(self.group1_indexes) else 0
            actual_idx = self.col_index_map[chosen_col_pixel]

            for slot in range(2):
                if self.bullet_y_positions[actual_idx, slot] == 0:
                    self.bullet_y_positions[actual_idx, slot] = enemy_y
                    break
        else:
            print("No free bullet slots available.")



    def update_bullet_positions(self):
        """
        Update positions of all bullets
        """
        bullet_speed = -0.00972  # Human bullet speed
        enemy_bullet_speed = 0.00277  # Enemy bullet speed

        # Clear lists of bullets on left and right sides
        self.bullets_left_side.clear()
        self.bullets_right_side.clear()

        # Update player bullets
        for i in self.playerb_used[:]:  # Copy list to avoid modification issues during iteration
            self.player_bullets[i][1] += bullet_speed
            # Remove out-of-range player bullets
            if self.player_bullets[i][1] > 1 or self.player_bullets[i][1] < 0:
                self.player_bullets[i] = [0, 0, None]
                self.playerb_used.remove(i)
                self.playerb_free.append(i)

        # Update enemy bullets
        for i in self.enemyb_used[:]:  # Copy list to avoid modification issues during iteration
            self.enemy_bullets[i][1] += enemy_bullet_speed 
            # Remove out-of-range enemy bullets
            if self.enemy_bullets[i][1] > 1.0 or self.enemy_bullets[i][1] < 0:
                self.enemy_bullets[i] = [0., 0.]
                self.enemyb_used.remove(i)
                self.enemyb_free.append(i)

        # Update bullet y positions for all slots, separate from enemy bullet updates
        for col in range(10):
            for slot in range(2):  # Assuming 2 slots per column
                if self.bullet_y_positions[col, slot] != 0:
                    self.bullet_y_positions[col, slot] += enemy_bullet_speed
                    if self.bullet_y_positions[col, slot] > 1.0 or self.bullet_y_positions[col, slot] < 0:
                        self.bullet_y_positions[col, slot] = 0  # Reset if out of bounds

        # Categorize bullets based on screen side
        for i in self.enemyb_used:
            bullet_screen_x = int(self.enemy_bullets[i][0] * self.screen_width)
            if bullet_screen_x < self.screen_width / 2:
                self.bullets_left_side.append([self.enemy_bullets[i][0], self.enemy_bullets[i][1]])
            else:
                self.bullets_right_side.append([self.enemy_bullets[i][0], self.enemy_bullets[i][1]])

        # Sort bullets by distance - only if needed
        self.sort_enemy_bullets_by_distance()


    def sort_enemy_bullets_by_distance(self):
        """
        Sorts enemy bullets by their Euclidean distance to the human spaceship.
        """

        #human position
        human_pos = np.array([self.human_position_x, self.human_position_y])
        left_bullets_only = np.copy(self.enemy_bullets)
        left_bullets_only[left_bullets_only[:, 0] > 0.5] = 0

        #Get x differences of only bullets on human side of the screen to the human spaceship
        differences = left_bullets_only - human_pos
        x_differences = differences[:, 0]
       
        #Get distance from human position to bullets
        distances = np.linalg.norm(left_bullets_only - human_pos, axis=1)

        # Encode direction within the distance
        directional_distances = distances * np.sign(x_differences)
        directional_distances[distances >= 0.9] = 0

        # Ensure 0.0314 has a negative sign and 0.331 has a positive sign- these are the leftmost and right most bullets on the human side of the screen
        directional_distances[(left_bullets_only[:, 0] == 0.0314)] = -abs(directional_distances[(left_bullets_only[:, 0] == 0.0314)])
        directional_distances[(left_bullets_only[:, 0] == 0.331)] = abs(directional_distances[(left_bullets_only[:, 0] == 0.331)])


        #Adjust so that anywhere that bullets don't exist is zero
        distances[distances >= 0.9] = 0

        #Updated this is distance to enemy bullets negative means left, positive means right
        self.human_distance_to_bullets = np.sort(directional_distances) 
        # Extract non-zero distances
        non_zero_distances = self.human_distance_to_bullets[self.human_distance_to_bullets != 0]

        # Create a new list with three items, filled with zeros initially
        self.small_distance_to_bullets = [0.0] *3

        # Populate the new list with non-zero distances, up to three items. There is a maxiumum of 3 bullets that can exist on human side at a time.
        for i in range(min(len(non_zero_distances), 3)):
            self.small_distance_to_bullets[i] = non_zero_distances[i]


    def check_bullet_collision(self, bullet1, bullet2, collision_distance=0.01):
        """
        Checks to see if two bullets have collided with one another
        """
        # Calculate the distance between two bullets.
        #TO DO: INCLUDE THIS IN CHECK COLLISIONS FUNCTION
        distance = ((bullet1[0] - bullet2[0]) ** 2 + (bullet1[1] - bullet2[1]) ** 2) ** 0.5
        return distance < collision_distance
    
    def check_collisions(self):
        """
        Checks to see if a bullets has collided with an agent's hitbox or if a bullet has collided with an enemy spaceship's hitbox
        """
        bullet_size = 0.01  # Assuming this is an approximation of bullet dimensions relative to screen size
        enemy_width = 50 / self.screen_width  # Enemy width adjusted for screen size
        enemy_height = 48 / self.screen_height  # Enemy height adjusted for screen size
        bullets_to_remove = []
        enemy_bullets_to_remove = []
        self.single_turn_score = {key:0 for key in self.single_turn_score}

        # Check collisions between player bullets and enemies
        for i, (x, y) in enumerate(zip(self.enemies_x, self.enemies_y)):
            if not self.enemies_active[i]:
                continue  # Skip inactive enemies

            # Create a rectangle for the enemy hitbox
            enemy_rect = pygame.Rect(
                (x - enemy_width / 2) * self.screen_width, 
                (y - enemy_height / 2) * self.screen_height,
                enemy_width * self.screen_width,
                enemy_height * self.screen_height
            )

            for index in self.playerb_used:
                bullet_x, bullet_y, shooter = self.player_bullets[index]
                # Create a rectangle for the bullet hitbox
                bullet_rect = pygame.Rect(
                    (bullet_x * self.screen_width)- 2, #checked using rectangle in render. Getting the top left and right points
                    (bullet_y  * self.screen_height) -6.5,#checked using rectangle in render. 
                    5, #width
                    15 #length
                )

                # Check if the bullet rectangle collides with the enemy rectangle
                if enemy_rect.colliderect(bullet_rect):
                    self.last_frame_score = np.copy(self.scores)
                    self.enemies_active[i] = False
                    if shooter == "Human":
                        #only count rewards if its from human
                        self.elimination_reward[shooter] += 5  # Add reward to shooter

                    elif shooter == "Nao":
                        if x < 0.5:
                            self.scores['Human'] += 10  # Add score to Human if Nao shot it on the left side
                            self.scores['Nao'] += 10  # Add score to Nao
                            self.nao_increasing_player_score[0]= 1
                            self.single_turn_score['Human'] = 10 #Increase of 10 points for Human for this turn

                        else:
                            self.scores['Shutter'] += 10  # Add score to Shutter if Nao shot it on the right side
                            self.scores['Nao'] += 10  # Add score to Nao
                            self.nao_increasing_player_score[1]= 1
                            self.single_turn_score['Shutter'] = 10 #Increase of 10 points for Shutter for this turn

                    self.scores[shooter] +=10
                    self.single_turn_score[shooter] +=10
                    self.enemies_hit_frames[i] = 0
                    bullets_to_remove.append(index)  # Mark bullet for removal

        # Check collisions between enemy bullets and players (Human, Shutter, Nao)
        for index in self.enemyb_used:
            bullet_x, bullet_y = self.enemy_bullets[index]
            bullet_rect = pygame.Rect(
                (bullet_x * self.screen_width)- 2, #checked using rectangle in render. Getting the top left and right points
                (bullet_y  * self.screen_height) -6.5,#checked using rectangle in render. 
                5, #width
                15 #length
            )

            # Define hit boxes for Human, Shutter, Nao
            for player_name, player_x, player_y, player_active in [
                ('Human', self.human_position_x, self.human_position_y, self.active['Human']),
                ('Shutter', self.shutter_position_x, self.shutter_position_y, self.active['Shutter']),
                ('Nao', self.nao_position_x, self.nao_position_y, self.active['Nao'])
            ]:
                if player_active:
                    player_rect = pygame.Rect(
                        (player_x - enemy_width / 2) * self.screen_width, 
                        (player_y - enemy_height / 2) * self.screen_height,
                        enemy_width * self.screen_width,
                        enemy_height * self.screen_height
                    )
                    # Check if any enemy bullet hits Human, Shutter, or Nao
                    if player_rect.colliderect(bullet_rect):
                        if player_name == 'Human':
                            self.elimination_reward[player_name] -= 100 #Penalty for human being hit by a bullet, make 
                            self.human_hit_count +=1
                            self.active[player_name] = False #This deactivates the human when it is hit. Moving it after the loop causes it to deactive when Nao or Shutter are hit
                        enemy_bullets_to_remove.append(index)

        # Remove bullets that hit an enemy
        for index in sorted(bullets_to_remove, reverse=True):
            self.player_bullets[index] = [0, 0, None]
            if index in self.playerb_used:
                self.playerb_used.remove(index)
            self.playerb_free.append(index)

        # Remove enemy bullets that hit players
        for index in sorted(enemy_bullets_to_remove, reverse=True):
            self.enemy_bullets[index] = [0, 0]
            self.bullet_y_positions[index] = 0  # Reset the y position in bullet_y_positions array
            if index in self.enemyb_used:
                self.enemyb_used.remove(index)
            self.enemyb_free.append(index)
        


    def enemy_respawn_handler(self):
        """
        Respawn enemies after being eliminated
        """
        #print('enemies hit frames', self.enemies_hit_frames)
        for i in range(len(self.enemies_hit_frames)):
            if self.enemies_hit_frames[i] >= 0:  # Enemy has been hit
                self.enemies_hit_frames[i] += 1
                if self.enemies_hit_frames[i] >= 300 :  # Respawn enemy after 50 frames
                    self.enemies_active[i] = True
                    self.enemies_hit_frames[i] = -1  # Reset frame counter

    def human_respawn_handler(self):
        """
        Respawn player agents (Nao, Human, Shutter) after being eliminated
        """
        for agent in self.agents:
            if self.active[agent] == False:
                self.active[agent] = True
                self.human_position_x = self.initial_human_position_x
                self.human_position_y = self.initial_human_position_y

    def render(self):
        bullet_size = .006

        # Clear the screen
        self.screen.fill((0, 0, 0))  # Black background

        # Draw human player if active
        if self.active['Human']:
            human_rect = self.ship_image.get_rect(center=(int(self.human_position_x * self.screen_width), int(self.human_position_y * self.screen_height)))
            self.screen.blit(self.ship_image, human_rect)
        # Draw shutter if active
        if self.active['Shutter']:
            shutter_rect = self.shutter_image.get_rect(center = (int(self.shutter_position_x * self.screen_width), int(self.shutter_position_y * self.screen_height)))
            self.screen.blit(self.shutter_image, shutter_rect)
        # Draw Nao in white if active
        if self.active['Nao']:
            nao_rect = self.nao_image.get_rect(center =(int(self.nao_position_x * self.screen_width), int(self.nao_position_y * self.screen_height)))
            self.screen.blit(self.nao_image, nao_rect)
        # Draw active enemies
        if self.group1_fast:
            for (x, y), active in zip(zip(self.enemies_x, self.enemies_y), self.enemies_active):
                if active:
                    if x >= 0.581:
                        # Use the second enemy image for enemies on the right half
                        enemy_rect = self.enemy_image4.get_rect(center=(int(x * self.screen_width), int(y * self.screen_height)))
                        self.screen.blit(self.enemy_image4, enemy_rect)
                    else:
                        # Use the first enemy image for other enemies
                        enemy_rect = self.enemy_image3.get_rect(center=(int(x * self.screen_width), int(y * self.screen_height)))
                        self.screen.blit(self.enemy_image3, enemy_rect)

        else:
            for (x, y), active in zip(zip(self.enemies_x, self.enemies_y), self.enemies_active):
                if active:
                    if x >= 0.581:
                        # Use the second enemy image for enemies on the right half
                        enemy_rect = self.enemy_image3.get_rect(center=(int(x * self.screen_width), int(y * self.screen_height)))
                        self.screen.blit(self.enemy_image3, enemy_rect)
                    else:
                        # Use the first enemy image for other enemies
                        enemy_rect = self.enemy_image4.get_rect(center=(int(x * self.screen_width), int(y * self.screen_height)))
                        self.screen.blit(self.enemy_image4, enemy_rect)
        for x, y, shooter in self.player_bullets[self.playerb_used]:
            bullet_rect = self.bullet_image.get_rect(center=(int(x * self.screen_width), int(y * self.screen_height)))
            self.screen.blit(self.bullet_image, bullet_rect)

        # Draw enemy bullets (if they are different, load and use another image)
        for x, y in self.enemy_bullets[self.enemyb_used]:
            enemy_bullet_rect = self.enemy_bullet_image.get_rect(center=(int(x * self.screen_width), int(y * self.screen_height)))
            self.screen.blit(self.enemy_bullet_image, enemy_bullet_rect)

        # Render player 1 score
        p1_score_text = f"YOUR SCORE: {int(self.scores['Human'])}"
        p1_score_surface = self.font.render(p1_score_text, True, (0, 255, 63))  # Green text
        p1_score_rect = p1_score_surface.get_rect(center=(self.screen_width / 2, 20))
        self.screen.blit(p1_score_surface, p1_score_rect)
        
        # Render player 2 score
        p2_score_text = f"SHUTTER SCORE: {int(self.scores['Shutter'])}"
        p2_score_surface = self.font.render(p2_score_text, True, (0, 255, 63))  # Green text
        p2_score_rect = p2_score_surface.get_rect(center=(self.screen_width / 2, 50))  # Adjust Y to not overlap with player 1 score
        self.screen.blit(p2_score_surface, p2_score_rect)

        # Render frame count- for testing 
        frame_screen = f"frame: {int(self.frame)}"
        frame_screen_surface = self.font.render(frame_screen, True, (0, 255, 63))  # Green text
        frame_screen_rect = frame_screen_surface.get_rect(topleft=(10, 50))  # Adjust Y to not overlap with player 1 score
        self.screen.blit(frame_screen_surface, frame_screen_rect)
        
        # Time
        remaining_time = 120 - int(self.time)  # 120 seconds equals 2 minutes
        minutes = remaining_time // 60
        seconds = remaining_time % 60
        time_text = f"TIMER {minutes:02}:{seconds:02}"
        time_surface = self.font.render(time_text, True, (255, 255, 255))  # Green text
        time_rect = time_surface.get_rect(topright=(self.screen_width - 10, 10))  # Adjust Y to not overlap with player 1 score
        self.screen.blit(time_surface, time_rect)

        # #Draw bullet rectangels
        # for index in self.enemyb_used:
        #     bullet_x, bullet_y = self.enemy_bullets[index]
        #     bullet_rect = pygame.Rect(
        #         (bullet_x * self.screen_width)- 2, 
        #         (bullet_y  * self.screen_height) -6.5,
        #         5,
        #         15
        #     )
        #     sur = (bullet_x *self.screen_width,bullet_y*self.screen_height)
        #     pygame.draw.rect(self.screen,'white',bullet_rect)

        

        # Update the display
        if self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        elif self.render_mode == 'human':
            pygame.display.flip()



