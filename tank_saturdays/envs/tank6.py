"""
Gym Environment for Tank Saturdays.

This environment simulates a battle between two tanks, with limited fuel,
bullets, and hit-points. The tanks start in opposite sides of a square
battlefield with randomly placed obstacles in the middle.

If a tank loses all hit-points or gasoline, it loses. If both tanks lose in the
same turn, it results in a draw. If the tanks collide, this also results in a
draw.

Every few turns, a box with extra fuel, bullets or HP might appear in the
battlefield. It will not appear on walls, tanks, or other boxes. Tanks have to
go over it in order to aquire the contents. Boxes are placed at the end of each
turn. Bullets pass over boxes without colliding with them.

Created by Pablo Talavante and Miguel Blanco.
"""


import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import typer
from collections import namedtuple, defaultdict
from recordclass import recordclass

class TankSaturdays(gym.Env):
    metadata = {
        'render.modes' : ['rgb_array', 'console'],
        'video.frames_per_second' : 30
    }

    def __init__(self):

        BATTLEFIELD_SIDE = 50  # Distance between battlefield corners
        TANK_SIDE = 5          # Size of the tanks
        GAS = 2000             # Ammount of fuel in each tank at the start
        CARTRIDGE = 100        # Ammount of bullets in each tank at the start
        HP = 3                 # Hit-points each tank has at the start
        BULLET_SPEED = 5       # Distance travelled by bullets each turn
        IDLE_COST = 1          # Minimum fuel consumed each turn
        MOVE_COST = 2          # Extra fuel consumed for moving in a turn
        N_WALLS = 10           # Number of random walls placed at the start
        WIDTH_WALLS = 2        # Width of the random walls
        MTB_BOXES = 50         # Mean Turns Between boxes with loot
        MAX_BOXES = 5          # Maximum number of boxes in the battlefield
        BOX_FUEL_P = 0.5       # Probability of a box having fuel
        BOX_BULLET_P = 0.3     # Probability of a box having bullets
        BOX_HP_P = 0.2         # Probability of a box having extra hit-point
        BOX_FUEL_AMNT = 200    # Amount of fuel in a fuel box
        BOX_BULLET_AMNT = 20   # Amount of bullets in a bullet box
        BOX_HP_AMNT = 1        # Amount of hit-points in a hit-point box

        # Game settings
        self.bf_side = BATTLEFIELD_SIDE
        self.tank_side = TANK_SIDE
        self.gas = GAS
        self.cartridge = CARTRIDGE
        self.HP = HP
        self.bullet_speed = BULLET_SPEED
        self.idle_cost = IDLE_COST
        self.move_cost = MOVE_COST
        self.n_walls = N_WALLS
        self.width_walls = WIDTH_WALLS
        self.length_walls = (self.tank_side, self.bf_side//2)
        self.mtb_boxes = MTB_BOXES
        self.max_boxes = MAX_BOXES
        self.box_probs = (BOX_FUEL_P, BOX_BULLET_P, BOX_HP_P)
        self.box_amounts = {"gas": BOX_FUEL_AMNT,
                            "bullet": BOX_BULLET_AMNT,
                            "hp": BOX_HP_AMNT}

        self.bf_size = np.array([self.bf_side, self.bf_side])
        self.tank_size = np.array([self.tank_side, self.tank_side])
        self.pad = self.tank_side//2

        self.action_space = spaces.Discrete(9)

        ram_size = (4*2 +
                    4*self.max_boxes +
                    4*(2*(self.bf_side//self.bullet_speed)+4) +
                    4*self.n_walls)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=10, shape=self.bf_size, dtype=np.int8),
            spaces.Box(low=-np.inf, high=np.inf,
                       shape=(ram_size,), dtype=np.int32)
            ))


        self.action_map = {  # Not actually used, just for reference
            "idle": 0,
            "move_up": 1,
            "move_right": 2,
            "move_down": 3,
            "move_left": 4,
            "shoot_up": 5,
            "shoot_right": 6,
            "shoot_down": 7,
            "shoot_left": 8,
            }

        # Named tuples and lists used for recording game object properties
        self.bullet = recordclass('Bullet', ['x', 'y', 'dx', 'dy'])
        self.wall = namedtuple('Wall', ['x0', 'y0', 'x1', 'y1'])
        self.tank = recordclass('Tank', ['x', 'y', 'gas', 'cartridge', 'HP'])
        self.dv_tuple = namedtuple('Velocities', ['dx', 'dy'])
        self.shoot_tuple = namedtuple('Shoot', ['x', 'y', 'dx', 'dy'])
        self.point = namedtuple('Point', ['x', 'y'])
        self.box = namedtuple('Box', ['x', 'y', 'content', 'amount'])

        # Map of actions to tank movement, defaulting to no movement
        self.v_actions = defaultdict(lambda: self.dv_tuple(0, 0))
        self.v_actions[1] = self.dv_tuple(0, -1)
        self.v_actions[2] = self.dv_tuple(1, 0)
        self.v_actions[3] = self.dv_tuple(0, 1)
        self.v_actions[4] = self.dv_tuple(-1, 0)

        # Map of actions to tank shots, defaulting to no shot
        self.s_actions = defaultdict(lambda: None)
        self.s_actions[5] = self.shoot_tuple(
            0, -self.pad-1, 0, -self.bullet_speed)
        self.s_actions[6] = self.shoot_tuple(
            self.pad+1, 0, self.bullet_speed, 0)
        self.s_actions[7] = self.shoot_tuple(
            0, self.pad+1, 0, self.bullet_speed)
        self.s_actions[8] = self.shoot_tuple(
            -self.pad-1, 0, -self.bullet_speed, 0)


        self.last_action_b = None
        self.last_action_w = None
        self.viewer = None
        self.hits = list()
        self.image = None
        self.ram = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        """Creates a numpy RandomState for use in other methods."""
        self.np_random, seed = seeding.np_random(
            seed if seed is not None else np.random.seed())
        return [seed]

    def reset(self):
        """Reset the field of play."""

        # Put tanks in opposite corners, with max gas, bullets and life
        self.black = self.tank(self.pad, self.pad,
            self.gas, self.cartridge, self.HP)
        self.white = self.tank(self.bf_side-self.pad-1, self.bf_side-self.pad-1,
            self.gas, self.cartridge, self.HP)

        # Reset bullets in play
        self.bullets = []
        self.new_bullets = []
        self.hits = []  # Merely for rendering purposes, but need to keep track

        # Put walls in random locations in the center of the battlefield
        self.wall_m = np.zeros(self.bf_size)
        self.walls = []
        for _ in range(self.n_walls):
            dir = self.np_random.randint(2)
            length = self.np_random.randint(*self.length_walls)
            x0 = self.np_random.randint(low = self.tank_side,
                high = self.bf_side-self.tank_side-length)
            y0 = self.np_random.randint(low = self.tank_side,
                high = self.bf_side-self.tank_side-self.width_walls)
            x1 = x0 + length
            y1 = y0 + self.width_walls
            if dir == 1:  # If vertical wall, transpose dimensions
                x0, x1, y0, y1 = y0, y1, x0, x1

            self.walls.append(self.wall(x0, y0, x1, y1))
            self.wall_m[x0:x1, y0:y1] = 1  # Matrix with ones in wall positions

        # Reset boxes
        self.boxes = []

        self.state = None
        return self._get_obs()

    def step(self, action_b, action_w=None):
        """Executes actions for each tank, simultaneously, and advances time."""

        self.last_action_b = action_b
        self.last_action_w = action_w

        # Get tank displacement from actions
        action_b_v = self.v_actions[action_b]
        action_w_v = self.v_actions[action_w]

        # Get tank shots from actions
        action_b_s = self.s_actions[action_b]
        action_w_s = self.s_actions[action_w]

        # Substract gas from tanks from basic functioning
        self.black.gas -= self.idle_cost
        self.white.gas -= self.idle_cost

        # Suppress movement when it would collide or if there is no gas left
        if self._wall_collision(self.black, action_b_v
            ) or self.black.gas < self.move_cost:
            action_b_v = self.v_actions[0]
        if self._wall_collision(self.white, action_w_v
            ) or self.white.gas < self.move_cost:
            action_w_v = self.v_actions[0]

        # Suppress shooting if cartridge is empty
        if self.black.cartridge <= 0:
            action_b_s = None
        if self.white.cartridge <= 0:
            action_w_s = None

        # Move tanks
        self._move_tanks(action_b_v, action_w_v)

        # Substract gas from tanks if they moved
        if action_b_v != self.v_actions[0]: self.black.gas -= self.move_cost
        if action_w_v != self.v_actions[0]: self.white.gas -= self.move_cost

        # Move bullets
        self._move_bullets()

        # Shoot new bullets
        self._shoot_bullets(action_b_s, action_w_s)

        # Substract bullets from cartridge if there was a shoot action
        if action_b_s: self.black.cartridge -= 1
        if action_w_s: self.white.cartridge -= 1

        # Check bullet-wall collisions and remove collided bullets
        self._bullet_wall_collisions()  # Before extend, due to bullet speed

        # Check bullet-tank collisions
        self.hits = []  # This is just for rendering, no effect on gameplay
        if self._bullet_tank_collision(self.black): self.black.HP -= 1
        if self._bullet_tank_collision(self.white): self.white.HP -= 1

        # Add new bullets to record of bullets in flight
        self.bullets.extend(self.new_bullets)

        # Check bullet-bullet collisions and remove collided bullets
        self._bullet_bullet_collisions()

        # Remove bullets out of battlefield
        self._remove_bullets()

        # Check collisions between tanks and end game as draw if they collided
        if self._tank_collision():
            return self._get_obs(), 0, True, {}

        # Check if tanks caught any boxes. Remove those boxes and add contents
        self._catch_boxes(self.black)
        self._catch_boxes(self.white)

        # End game in case of death(s)
        if (self.black.HP <= 0) and (self.white.HP <= 0):  # Draw
            return self._get_obs(), 0, True, {}
        elif self.black.HP <= 0:  # White tank wins
            return self._get_obs(), -1, True, {}
        elif self.white.HP <= 0:  # Black tank wins
            return self._get_obs(), 1, True, {}

        # If tank runs out of gas, they lose
        if (self.black.gas <= 0) and (self.white.gas <= 0):
            return self._get_obs(), 0, True, {}
        elif self.black.gas <= 0:  # White tank wins
            return self._get_obs(), -1, True, {}
        elif self.white.gas <= 0:  # Black tank wins
            return self._get_obs(), 1, True, {}

        # Attempt to place a box with extra consumables
        if (self.np_random.uniform() < 1./self.mtb_boxes):
            box_type = self.np_random.choice(("gas", "bullet", "hp"),
                p=self.box_probs)
            self._place_box(box_type)

        # By default, no rewards are given and game continues
        return self._get_obs(), 0, False, {}

    def _get_obs(self):
        """Compose observation for the agent."""

        # Background is 0
        self.image = np.zeros(self.bf_size, dtype=np.int8)

        # Black tank is 1, white tank is 2
        self.image[self.black.x-self.pad:self.black.x+self.pad+1,
                   self.black.y-self.pad:self.black.y+self.pad+1] = 1
        self.image[self.white.x-self.pad:self.white.x+self.pad+1,
                   self.white.y-self.pad:self.white.y+self.pad+1] = 2

        # Boxes are 3 for gas, 4 for bullets and 5 for hp
        box_num = {"gas": 0, "bullet": 1, "hp": 2}
        for box in self.boxes:
            self.image[box.x, box.y] = box_num[box.content] + 3

        # Bullets. This might overwrite boxes in the image (but not in ram)
        for bullet in self.bullets:
            if bullet.dy < 0:    # Bullet moving up is 6
                self.image[bullet.x, bullet.y] = 6
            elif bullet.dx > 0:  # Bullet moving right is 7
                self.image[bullet.x, bullet.y] = 7
            elif bullet.dy > 0:  # Bullet moving down is 8
                self.image[bullet.x, bullet.y] = 8
            elif bullet.dx < 0:  # Bullet moving left is 9
                self.image[bullet.x, bullet.y] = 9

        # Walls
        for wall in self.walls:  # Walls are 10
            self.image[wall.x0:wall.x1, wall.y0:wall.y1] = 10


        # Gather variables from tanks, bullets, boxes, walls
        tank_info = [self.black.x,
                     self.black.y,
                     self.black.gas,
                     self.black.cartridge,
                     self.black.HP,
                     self.white.x,
                     self.white.y,
                     self.white.gas,
                     self.white.cartridge,
                     self.white.HP,
                     ]

        box_info = []
        for i in range(self.max_boxes):
            if i < len(self.boxes):
                box = self.boxes[i]
                box_info.extend([box.x,
                                 box.y,
                                 box_num[box.content],
                                 box.amount])
            else:
                box_info.extend([-1,-1,-1,-1])  # Invalid box out of bounds

        max_possible_bullets = 2*(self.bf_side//self.bullet_speed) + 4
        bullet_info = []
        for i in range(max_possible_bullets):
            if i < len(self.bullets):
                bullet = self.bullets[i]
                bullet_info.extend([bullet.x,
                                    bullet.y,
                                    bullet.dx,
                                    bullet.dy])
            else:
                bullet_info.extend([-1,-1,0,0])  # Unmoving bullet out of bounds

        wall_info = []
        for wall in self.walls:
            wall_info.extend([wall.x0,
                              wall.y0,
                              wall.x1,
                              wall.y1])

        self.ram = np.array(tank_info + box_info + bullet_info + wall_info,
                   dtype=np.int32)
        self.state = (self.image, self.ram)
        return self.state

    def _move_tanks(self, action_b_v, action_w_v):
        """Execute all tank movements."""
        self.black.x += action_b_v.dx
        self.black.y += action_b_v.dy
        self.white.x += action_w_v.dx
        self.white.y += action_w_v.dy

    def _move_bullets(self):
        """Displace all bullets in flight."""
        for i, bullet in enumerate(self.bullets):
            self.bullets[i].x += bullet.dx
            self.bullets[i].y += bullet.dy

    def _shoot_bullets(self, action_b_s, action_w_s):
        """Check if any tank tried to shoot and create the bullet if so."""
        self.new_bullets = []
        if action_b_s is not None:
            self.new_bullets.append(self.bullet(
                self.black.x+action_b_s.x, self.black.y+action_b_s.y,
                action_b_s.dx, action_b_s.dy))
        if action_w_s is not None:
            self.new_bullets.append(self.bullet(
                self.white.x+action_w_s.x, self.white.y+action_w_s.y,
                action_w_s.dx, action_w_s.dy))

    def _remove_bullets(self):
        """Removes bullets that went out of the battlefield."""
        # Implemented by keeping bullets inside bounds of battlefield
        self.bullets = list(filter(lambda b: (b.x>=0) and (b.x<self.bf_side),
                                         self.bullets))
        self.bullets = list(filter(lambda b: (b.y>=0) and (b.y<self.bf_side),
                                         self.bullets))

    def _wall_collision(self, tank, action_v):
        """Returns True if the tank would collide with a wall or border if it
        acted on given action.
        """
        dx = action_v.dx
        dy = action_v.dy

        if (dx==0) and (dy==0):  # No need to check this case, can't collide
            return False

        # Make a matrix with ones in positions occupied by the tank after move
        tank_m = np.zeros(self.bf_size + np.array([2,2]))
        tank_m[tank.x+1-self.pad + dx:tank.x+1+self.pad+1 + dx,
               tank.y+1-self.pad + dy:tank.y+1+self.pad+1 + dy] = 1

        # Make a matrix with ones in positions occupied by walls and borders
        wall_m = np.zeros(self.bf_size + np.array([2,2]))
        wall_m[[0,-1],:], wall_m[:, [0,-1]] = 1, 1  # Walls around battlefield
        for wall in self.walls:
            wall_m[wall.x0+1:wall.x1+1, wall.y0+1:wall.y1+1] = 1

        # Multiply element-wise and sum
        collisions = np.sum(wall_m * tank_m)

        return collisions > 0

    def _bullet_wall_collisions(self):
        """Remove all bullets that collided in the last step."""
        self.bullets = list(filter(self._bw_no_collision, self.bullets))
        self.new_bullets = list(filter(
            self._new_bw_no_collision, self.new_bullets))

    def _bw_no_collision(self, bullet):
        """Checks collitions between a given bullet and the walls."""
        x = bullet.x
        y = bullet.y
        for i in range(self.bullet_speed):
            if self._point_wall_collision(x, y):
                return False  # This means it did collide
            x -= np.sign(bullet.dx)  # Retract bullet steps this turn
            y -= np.sign(bullet.dy)
        return True

    def _new_bw_no_collision(self, bullet):
        """Checks collitions between a new bullet and the walls."""
        return not self._point_wall_collision(bullet.x, bullet.y)

    def _point_wall_collision(self, x, y):
        """See if a coordinate is part of the walls."""
        try:
            return self.wall_m[x, y] == 1
        except:
            return False  # In case inputed point is out of array bounds

    def _bullet_tank_collision(self, tank):
        """Check if the tank was hit by any bullets."""
        hit = False
        for bullet in self.new_bullets.copy():
            if self._point_in_tank(tank, bullet.x, bullet.y):
                self.new_bullets.remove(bullet)  # Remove the bullet if it hit
                self.hits.append((self.point(bullet.x, bullet.y)))
                hit = True
        for bullet in self.bullets.copy():
            for i in range(self.bullet_speed):  # Look at trail of bullet
                x = bullet.x - i*np.sign(bullet.dx)
                y = bullet.y - i*np.sign(bullet.dy)
                if self._point_in_tank(tank, x, y):
                    self.bullets.remove(bullet)  # Remove the bullet if it hit
                    self.hits.append((self.point(x, y)))
                    hit = True
                    break  # Can't return cause other bullets might hit too
        return hit

    def _point_in_tank(self, tank, x, y):
        """Return true if a point belongs to a tank."""
        return ((x >= tank.x-self.pad) and (x <= tank.x+self.pad) and
                (y >= tank.y-self.pad) and (y <= tank.y+self.pad))

    def _bullet_bullet_collisions(self):
        """Remove all bullets that share same position."""
        keep_bullets = set()
        for i, bullet_1 in enumerate(self.bullets):
            collision = False
            for j, bullet_2 in enumerate(self.bullets):
                if (bullet_1.x == bullet_2.x) and (bullet_1.y == bullet_2.y):
                    if j != i:
                        collision = True
            if not collision:
                keep_bullets.add(i)
        self.bullets = [self.bullets[i] for i in keep_bullets]

    def _tank_collision(self):
        """Returns True if the tanks collide in the current position."""
        x_collide = ((self.white.x+self.pad+1 > self.black.x-self.pad) and
                     (self.black.x+self.pad+1 > self.white.x-self.pad))
        y_collide = ((self.white.y+self.pad+1 > self.black.y-self.pad) and
                     (self.black.y+self.pad+1 > self.white.y-self.pad))
        return x_collide and y_collide

    def _catch_boxes(self, tank):
        # Determine which boxes were caught
        left_boxes = []
        for box in self.boxes:
            if self._point_in_tank(tank, box.x, box.y):
                if box.content == "gas":
                    tank.gas += box.amount
                elif box.content == "bullet":
                    tank.cartridge += box.amount
                elif box.content == "hp":
                    tank.HP += box.amount
            else:
                left_boxes.append(box)
        self.boxes = left_boxes

    def _place_box(self, box_type):
        """Place a box of consumables on a randomly chosen free spot."""

        # If already at max boxes, remove the one oldest one
        if len(self.boxes) == self.max_boxes:
            del self.boxes[0]

        # Find a free spot for the box
        free_spots = np.ones(self.bf_size, dtype=np.bool)
        free_spots[self.black.x-self.pad:self.black.x+self.pad+1,
                   self.black.y-self.pad:self.black.y+self.pad+1] = False
        free_spots[self.white.x-self.pad:self.white.x+self.pad+1,
                   self.white.y-self.pad:self.white.y+self.pad+1] = False
        for wall in self.walls:
            free_spots[wall.x0:wall.x1, wall.y0:wall.y1] = False

        free_spots_idx = list(np.argwhere(free_spots))
        idx_choice = self.np_random.choice(len(free_spots_idx))
        spot_x, spot_y = free_spots_idx[idx_choice]

        # Define the box
        box = self.box(spot_x, spot_y, box_type, self.box_amounts[box_type])

        # Place the box
        self.boxes.append(box)

    def render(self, mode='console'):

        if mode == 'rgb_array':
            pass

        if mode == "console":
            # Background
            self.render_m = np.full(self.bf_size, '·')

            # Tanks
            self.render_m[self.black.x-self.pad:self.black.x+self.pad+1,
                          self.black.y-self.pad:self.black.y+self.pad+1] = '■'
            self.render_m[self.white.x-self.pad:self.white.x+self.pad+1,
                          self.white.y-self.pad:self.white.y+self.pad+1] = '□'

            # Boxes
            box_char = {"gas": "F", "bullet": "B", "hp": "H"}
            for box in self.boxes:
                self.render_m[box.x, box.y] =  box_char[box.content]

            # Bullets
            for bullet in self.bullets:
                if np.abs(bullet.dx) == self.bullet_speed:
                    self.render_m[bullet.x, bullet.y] = '|'
                elif np.abs(bullet.dy) == self.bullet_speed:
                    self.render_m[bullet.x, bullet.y] = '—'

            # Walls
            for wall in self.walls:
                self.render_m[wall.x0:wall.x1, wall.y0:wall.y1] = 'X'

            # Hits
            for hit in self.hits:
                self.render_m[hit.x, hit.y] = "o"

            # Print to console
            for row in self.render_m.tolist():
                print(" ".join(row))
            print(f"Player   Gas   Bullets  HP")
            print(f"Black   {self.black.gas:4}       {self.black.cartridge:3}"
                  f"   {self.black.HP}")
            print(f"White   {self.white.gas:4}       {self.white.cartridge:3}"
                  f"   {self.white.HP:1}")
            typer.clear()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
