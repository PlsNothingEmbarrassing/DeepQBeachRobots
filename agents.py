import mesa

DEALT_WITH = 100
PICKED = 1
UNPICKED = 0

START = 0
MOVING_TO_AREA = 1
BUSY = 2
PICKING = 3
FULL = 4
EMPTYING = 5
PICKING = 6
EVADING = 7
FINISHED = 8

SEARCHING = 9
CHARGING = 10


CHARGE = 150

LC_CAPACITY = 30

GRID_LIMIT = 49

BIN_LOC = (49, 0)
CHARGING_STATION = (0, 0)


class SmallRubbish(mesa.Agent):
    def __init__(self, id, pos, model, state=UNPICKED, weight=1):
        super().__init__(id, model)
        self.x, self.y = pos
        self.state = state
        self.id = id
        self.model = model
        self.weight = weight

    def __str__(self):
        return "Small Rubbish"


class Obstacle(mesa.Agent):
    def __init__(self, id, model, pos):
        super().__init__(id, model)
        self.x, self.y = pos
        self.id = id
        self.model = model

    def __str__(self):
        return "Obstacle"


class Robot(mesa.Agent):
    def __init__(self, unique_id, model, pos, state=START, capacity=None, speed=None):
        super().__init__(unique_id, model)
        self.id = unique_id
        self.model = model
        self.x, self.y = pos
        self.state = state
        self.capacity = capacity
        self.speed = speed
        self.next_x, self.next_y = self.x, self.y
        self.payload = []

    @property
    def isFull(self):
        return self.capacity == 0

    @property
    def isBusy(self):
        return self.state == BUSY

    def wait(self):
        self.next_x = self.x
        self.next_y = self.y

    def advance(self):
        # Advances robot position
        self.x = self.next_x
        self.y = self.next_y

    def move(self):
        self.x = self.next_x
        self.y = self.next_y

    def get_payload_agent(self):
        all_agents = self.model.schedule.agents
        payload_agents = []
        for agent in all_agents:
            if agent.id not in self.payload:
                pass
            else:
                payload_agents.append(agent)
        return payload_agents


class LCRobot(Robot):
    def __init__(
        self,
        id,
        pos,
        model,
    ):
        super().__init__(id, model, pos)
        self.id = id
        self.x, self.y = pos
        self.model = model
        self.next_x, self.next_y = self.x, self.y
        # self.state = state
        # self.capacity = capacity
        self.payload = []

    @property
    def isFull(self):
        return super().isFull

    def advance(self):
        self.model.grid.move_agent(self, (self.next_x, self.next_y))
        self.x = self.next_x
        self.y = self.next_y

    def wait(self):
        return super().wait()

    def deliberate(self):
        if self.state == EVADING:
            return "evade_obstacle"
        elif self.state == START:
            self.state = MOVING_TO_AREA
            return "move_to_area"
        elif self.isFull and self.pos != BIN_LOC:
            return "move_to_bin"
        elif self.isFull and self.pos == BIN_LOC:
            return "emptying"

        elif self.state == MOVING_TO_AREA:
            return "move_to_area"
        elif self.state == BUSY:
            return "move_through_area"
        elif self.state == PICKING:
            return "pick"

        elif self.state == FINISHED:
            return "finished"

    def step(self):
        action = self.deliberate()
        if action == "move_to_area":
            self.move_to_cleaning_area()
            self.move_payload()
            self.advance()
        elif action == "move_through_area":
            self.move_through_area()
            self.move_payload()
            self.advance()
        elif action == "pick":
            self.pick_rubbish()
        elif action == "move_to_bin":
            self.move_to_bin()
            self.move_payload()
            self.advance()
        elif action == "emptying":
            self.drop_off()
        elif action == "finished":
            self.wait()
        elif action == "evade_obstacle":
            self.wait()

    def move_payload(self):
        payload_agents = self.get_payload_agent()
        for agent in payload_agents:
            agent.model.grid.move_agent(agent, (self.next_x, self.next_y))
            agent.pos = self.next_x, self.next_y

    # def move_to_bin(self):
    #     target = BIN_LOC
    #     if self.pos != target:
    #         target_x, target_y = target
    #         dx = target_x - self.x
    #         dy = target_y - self.y
    #         if abs(dx) > 0:
    #             self.next_x += dx // abs(dx)
    #         if abs(dy) > 0:
    #             self.next_y += dy // abs(dy)
    #     else:
    #         self.state = EMPTYING
    #         self.drop_off()

    def pick_rubbish(self):
        # Used to get the litter at current position with radius = 0
        litters = self.model.grid.get_neighbors(
            self.pos, moore=False, include_center=True, radius=0
        )

        if litters != None:
            for litter in litters:
                # Check if agent is an Small rubbish instance and has not been picked
                if isinstance(litter, SmallRubbish) and litter.state == UNPICKED:
                    # print("litter picked up")
                    self.payload.append(litter.id)
                    litter.state = PICKED

                    self.modify_capacity(litter.weight)
                    self.state = BUSY
                    if self.isFull:
                        # print("emptying")
                        self.x, self.start_y = self.pos
                        # Decrement start_y so that no rows are missed
                        self.start_y -= 1

        else:
            pass

    def check_for_rubbish(self):
        litters = self.model.grid.get_neighbors(
            self.pos, moore=False, include_center=True, radius=0
        )
        if litters:
            for litter in litters:
                if isinstance(litter, SmallRubbish) and litter.state == UNPICKED:
                    self.state = PICKING
                    return True
        return False

    # def check_capacity(self, litter_item) -> bool:
    #     return self.capacity - litter_item.weight > 0

    # def modify_capacity(self, weight):
    #     self.capacity -= weight

    def check_cell(self, pos):
        x, y = pos
        if (x > 49) or (y > 49):
            return
        neighbors = self.model.grid.get_neighbors(
            pos, moore=False, include_center=True, radius=0
        )
        if neighbors:
            for neighbor in neighbors:
                if isinstance(neighbor, Obstacle):
                    return "Obstacle"
                elif isinstance(neighbor, Robot) and neighbor != self:
                    return "Robot"
        else:
            return "advance"
        return "advance"

    def get_payload_agent(self):
        return super().get_payload_agent()

    def drop_off(self):
        print("Emptying")
        self.direction *= -1
        payload_items = self.get_payload_agent()
        for item in payload_items:
            item.state = DEALT_WITH
        self.payload = []
        # self.capacity = LC_CAPACITY
        print("Rubbish droppped off")
