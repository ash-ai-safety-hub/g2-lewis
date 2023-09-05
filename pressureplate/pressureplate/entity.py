import numpy as np


class Entity:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


class GridAgent(Entity):
    def __init__(self, id, x, y):
        super().__init__(id, x, y)
        self.escaped = False
        self.crushed = False
        self.treasure = 0

    def take_action(self, action, env):

        proposed_pos = [self.x, self.y]

        # Up
        if action == 0:
            proposed_pos[1] -= 1
            if not self._detect_collision(env, proposed_pos):
                self.y -= 1

        # Down
        elif action == 1:
            proposed_pos[1] += 1
            if not self._detect_collision(env, proposed_pos):
                self.y += 1

        # Left
        elif action == 2:
            proposed_pos[0] -= 1
            if not self._detect_collision(env, proposed_pos):
                self.x -= 1

        # Right
        elif action == 3:
            proposed_pos[0] += 1
            # if not env._detect_collision(proposed_pos):
            if not self._detect_collision(env, proposed_pos):
                self.x += 1

        # Noop
        else:
            pass

        agent_pos = [self.x, self.y]

        # Update treasure collected.
        # TODO update so that stepping on plate doesn't increment treasure.
        goals_pos = [[goal.x, goal.y] for goal in env.goals if not goal.achieved]
        plates_pos = [[plate.x, plate.y] for plate in env.plates if not plate.ever_pressed]
        for goal_pos in goals_pos:
            if agent_pos == goal_pos:
                self.treasure += 10000 * env.gamma**env.timestep
        for plate_pos in plates_pos:
            if agent_pos == plate_pos:
                self.treasure += 100 * env.gamma**env.timestep

        # Check if escaped
        escapes_pos = [[escape.x, escape.y] for escape in env.escapes]
        if np.any([agent_pos == escape_pos for escape_pos in escapes_pos]):
            self.escaped = True


    def _detect_collision(self, env, proposed_position):
        """Check for collision with (1) grid edge, (2) walls, (3) closed doors, or (4) other agents."""

        # Grid edge
        if np.any([
            proposed_position[0] < 0,
            proposed_position[1] < 0,
            proposed_position[0] >= env.grid_size[1],
            proposed_position[1] >= env.grid_size[0]
        ]):
            return True

        # Walls
        for wall in env.walls:
            if proposed_position == [wall.x, wall.y]:
                return True

        # Closed Door
        for door in env.doors:
            if not door.open:
                if proposed_position == [door.x, door.y]:
                    return True

        # Other agents
        for agent in env.agents:
            if proposed_position == [agent.x, agent.y]:
                return True

        return False

class IPDAgent(Entity):
    def __init__(self, id, x, y):
        super().__init__(id, x, y)

    def take_action(self, action, env):
        # Lie
        if action == 0:
            self.y = 0

        # Confess
        elif action == 1:
            self.y = 1

class MarketAgent(Entity):
    def __init__(self, id, x, y):
        super().__init__(id, x, y)

    def take_action(self, action, env):
        self.y = self.id + 2
        self.x = action

class Plate(Entity):
    def __init__(self, id, x, y):
        super().__init__(id, x, y)
        self.pressed = False
        self.ever_pressed = False


class Door(Entity):
    def __init__(self, id, x, y):
        super().__init__(id, x, y)
        self.open = False


class Wall(Entity):
    def __init__(self, id, x, y):
        super().__init__(id, x, y)


class Goal(Entity):
    def __init__(self, id, x, y):
        super().__init__(id, x, y)
        self.achieved = False

class Escape(Entity):
    def __init__(self, id, x, y):
        super().__init__(id, x, y)
