from .agents import LCRobot, SmallRubbish, Obstacle
import mesa
import random
import itertools
import math

class Beach(mesa.Model):
    def __init__(
        self,
        n_lc_robots,        
        n_obstacles,
        n_small_rubbish,        
        width,
        height,
    ):
        self.n_lc_robots = n_lc_robots        
        self.n_obstacles = n_obstacles
        self.n_small_rubbish = n_small_rubbish                
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.grid = mesa.space.MultiGrid(width, height, False)
        self.place_agents()
        self.running = True
        

    def place_agents(self):
        all_spaces = [
            (x, y) for x in range(self.grid.width) for y in range(self.grid.height)
        ]
        # Use counter to increment ids based on num agents
        counter = itertools.count()

        lc_r_positions = self.find_centre_positions()
        available_spaces = list(filter(lambda x:x not in lc_r_positions, all_spaces))
        search_spaces = self.assign_cleaning_area()
        # Creating and placing LC robots
        for i in range(self.n_lc_robots):  
            if lc_r_positions:          
                u_id = next(counter)
                x, y = lc_r_positions.pop()
                agent = LCRobot(u_id, (x, y), self, target_area=search_spaces[i])
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)
                #print("Placed LC at postion: " + str((x,y)))
            
        

        
        # Using shuffle so agent placement should be different each time
        random.shuffle(available_spaces)        

        # Creating and placing obstacles
        for i in range(self.n_obstacles):
            if available_spaces:
                u_id = next(counter)
                x, y = available_spaces.pop()
                agent = Obstacle(u_id, self, (x, y))
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)                
                #print("Placed obstacle at position: " + str((x,y)))
            else:
                print("No available spaces to place agents")
        

            # Creating and placing large rubbish
        for i in range(self.n_small_rubbish):
            if available_spaces:
                u_id = next(counter)
                x, y = available_spaces.pop()
                agent = SmallRubbish(u_id, (x, y), self)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)
                #print("Placed SR at position: " + str((x,y)))
            else:
                print("No available spaces to place agents")
                
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
        self.schedule.step()