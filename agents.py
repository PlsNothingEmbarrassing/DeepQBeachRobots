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
EVADING = 6
FINISHED = 7

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
        target_area=None,
        capacity=LC_CAPACITY,
    ):
        super().__init__(
            id,
            model,
            pos,
            capacity=capacity,
        )
        self.id = id
        self.x, self.y = pos
        self.model = model
        self.next_x, self.next_y = self.x, self.y
        self.target_area = target_area
        self.payload = []
        self.action = None

    @property
    def isFull(self):
        return super().isFull

    def advance(self):
        target = (self.next_x, self.next_y)
        if target != self.pos:
            self.model.grid.move_agent(self, target)
        self.x = self.next_x
        self.y = self.next_y

    def wait(self):
        return super().wait()

    # def deliberate(self):
    #     if self.state == EVADING:
    #         return "evade_obstacle"
    #     elif self.state == START:
    #         self.state = MOVING_TO_AREA
    #         return "move_to_area"
    #     elif self.isFull and self.pos != BIN_LOC:
    #         return "move_to_bin"
    #     elif self.isFull and self.pos == BIN_LOC:
    #         return "emptying"

    #     elif self.state == MOVING_TO_AREA:
    #         return "move_to_area"
    #     elif self.state == BUSY:
    #         return "move_through_area"
    #     elif self.state == PICKING:
    #         return "pick"

    #     elif self.state == FINISHED:
    #         return "finished"

    def step(self):
        action = self.action
        if action == "move_up":
            self.move_up()
            self.move_payload()
            self.advance()
        elif action == "move_down":
            self.move_down()
            self.move_payload()
            self.advance()
        elif action == "move_left":
            self.move_left()
            self.move_payload()
            self.advance()
        elif action == "move_right":
            self.move_right()
            self.move_payload()
            self.advance()
        elif action == "pick":
            self.pick_rubbish()
            self.move_payload()
            self.advance()

    def move_payload(self):
        payload_agents = self.get_payload_agent()
        for agent in payload_agents:
            agent.model.grid.move_agent(agent, self.pos)
            agent.x, agent.y = self.pos

    def _set_move(self, dx, dy):
        target = (self.x + dx, self.y + dy)
        if self.check_cell(target) in ("Obstacle", "Robot"):
            self.wait()
            return False
        if not self.model.grid.out_of_bounds(target):
            self.next_x, self.next_y = target
            return True
        self.wait()
        return False

    def move_up(self):
        return self._set_move(0, 1)

    def move_down(self):
        return self._set_move(0, -1)

    def move_left(self):
        return self._set_move(-1, 0)

    def move_right(self):
        return self._set_move(1, 0)

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

        picked_any = False
        if litters is not None:
            for litter in litters:
                # Check if agent is an Small rubbish instance and has not been picked
                if isinstance(litter, SmallRubbish) and litter.state == UNPICKED:
                    # print("litter picked up")
                    if self.capacity is not None and self.capacity < litter.weight:
                        continue
                    self.payload.append(litter.id)
                    litter.state = PICKED
                    picked_any = True
                    if self.capacity is not None:
                        self.capacity -= litter.weight
                    self.state = BUSY
                    if self.isFull:
                        self.x, self.start_y = self.pos
                        self.start_y -= 1

        return picked_any

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
        if self.model.grid.out_of_bounds(pos):
            return "Boundary"
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
        payload_items = self.get_payload_agent()
        for item in payload_items:
            item.state = DEALT_WITH
        self.payload = []
        self.capacity = LC_CAPACITY

    def set_action(self, action):
        self.action = action
