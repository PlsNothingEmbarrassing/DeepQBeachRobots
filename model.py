import mesa
import itertools
import math

from .agents import LC_CAPACITY, LCRobot, SmallRubbish, Obstacle

class Beach(mesa.Model):
    def __init__(
        self,
        n_lc_robots,        
        n_obstacles,
        n_small_rubbish,        
        width,
        height,
        seed=None,
        robot_capacity=None,
        randomize_robot_start=False,
    ):
        counts = {
            "n_lc_robots": n_lc_robots,
            "n_obstacles": n_obstacles,
            "n_small_rubbish": n_small_rubbish,
        }
        integers = {**counts, "width": width, "height": height}
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integers.values()
        ):
            raise ValueError("grid dimensions and agent counts must be integers")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if any(value < 0 for value in counts.values()):
            raise ValueError("agent counts must be non-negative")
        if n_lc_robots == 0:
            raise ValueError("at least one robot is required")
        if sum(counts.values()) > width * height:
            raise ValueError("the grid does not have enough cells for all agents")
        if robot_capacity is not None and (
            not isinstance(robot_capacity, int) or robot_capacity < 0
        ):
            raise ValueError("robot_capacity must be a non-negative integer")
        if not isinstance(randomize_robot_start, bool):
            raise ValueError("randomize_robot_start must be a boolean")
        super().__init__(rng=seed)
        self.n_lc_robots = n_lc_robots        
        self.n_obstacles = n_obstacles
        self.n_small_rubbish = n_small_rubbish                
        self.robot_capacity = LC_CAPACITY if robot_capacity is None else robot_capacity
        self.randomize_robot_start = randomize_robot_start
        self.grid = mesa.space.MultiGrid(width, height, False)
        self.place_agents()
        self.running = True
        

    def place_agents(self):
        all_spaces = [
            (x, y) for x in range(self.grid.width) for y in range(self.grid.height)
        ]
        # Use counter to increment ids based on num agents
        counter = itertools.count()

        lc_r_positions = (
            self.random.sample(all_spaces, self.n_lc_robots)
            if self.randomize_robot_start
            else self.find_centre_positions()
        )
        robot_spaces = set(lc_r_positions)
        available_spaces = [space for space in all_spaces if space not in robot_spaces]
        search_spaces = self.assign_cleaning_area()
        # Creating and placing LC robots
        for i in range(self.n_lc_robots):
            u_id = next(counter)
            x, y = lc_r_positions.pop()
            agent = LCRobot(
                u_id,
                (x, y),
                self,
                target_area=search_spaces[i],
                capacity=self.robot_capacity,
            )
            self.grid.place_agent(agent, (x, y))
            
        

        
        # Using shuffle so agent placement should be different each time
        self.random.shuffle(available_spaces)

        # Creating and placing obstacles
        for _ in range(self.n_obstacles):
            u_id = next(counter)
            x, y = available_spaces.pop()
            agent = Obstacle(u_id, self, (x, y))
            self.grid.place_agent(agent, (x, y))
        # Creating and placing small rubbish
        for _ in range(self.n_small_rubbish):
            u_id = next(counter)
            x, y = available_spaces.pop()
            agent = SmallRubbish(u_id, (x, y), self)
            self.grid.place_agent(agent, (x, y))
                
    def assign_cleaning_area(self):
        ca_width = math.ceil(self.grid.width / self.n_lc_robots)
        ca_height = self.grid.height
        search_spaces = []
        for i in range(self.n_lc_robots):
            # Get starting positions for each robot search area
            start_x = (i*ca_width)
            start_y = 0
            # Calculating end positions
            end_x = min(start_x + ca_width, self.grid.width - 1)
            end_y = min(start_y + ca_height, self.grid.height - 1)
            search_space = (start_x, start_y, end_x, end_y)
            search_spaces.append(search_space)
        return search_spaces
    def find_centre_positions(self):
        # Placing lc robots in a circle around centre so they are not placed on top of each other
        centre_x = self.grid.width // 2
        centre_y = self.grid.height // 2
        positions = []
        for i in range(self.n_lc_robots):
            angle = 2* math.pi *( i / self.n_lc_robots)
            offset_x = int(centre_x + math.cos(angle) * (self.grid.width // 4))
            offset_y = int(centre_y + math.sin(angle) * (self.grid.height// 4))
            # Make sure that agents are within grid
            offset_x = max(0, min(self.grid.width-1, offset_x))
            offset_y = max(0, min(self.grid.height-1, offset_y))
            positions.append((offset_x, offset_y))
        return positions
        
    def step(self):
        self.agents.shuffle_do("step")
