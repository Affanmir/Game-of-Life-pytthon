import time
from tkinter import *
import random
import math
# An implementation of Conway's Game of Life
#
# Adapted from code by Abdullah Zafar


class Config:
    """Config class.

    Contains game configurations .
    """

    # Some starting configurations.
    glider = [(20, 40), (21, 40), (22, 40),
              (22, 41), (21, 42)]  # simple glider
    dense = [(21, 40), (21, 41), (21, 42), (22, 41),
             (20, 42)]  # explodes into a dense network
    oscillator = [(1, 4), (2, 4), (3, 4)]  # oscillator
    block = [(4, 4), (5, 4),
             (4, 5), (5, 5)]  # Block

    def __init__(self) -> None:
        """Provides a default configuration.

        Args:
        - self: automatic object reference.

        Returns:
        none
        """
        # ===== Life parameters
        self.start = Config.glider  # starting shape
        self.rounds = 5000  # number of rounds of the game

        # ===== Animation parameters
        self.animate: bool = False  # switch animation on or off
        # Screen dimensions
        self.width: int = 800
        self.height: int = 800
        # HU colors
        self.bg_color = '#e6d19a'
        self.cell_color = '#580f55'
        # Cell size. Cells are drawn at resolution CELL_SIZE x CELL_SIZE pixels.
        self.cell_size: int = 10
        # Animation speed. Positive integers, bigger is faster animation.
        self.speed: int = 1


w = 32


class LinearSet:
    def __init__(self, state: [(int, int)]) -> None:
        self.dela = ""
        self.d = 1
        self.t = [None]*2**self.d
        self.q = 0
        self.n = 0
        self.z = random.randrange(1, 99, 2)
        for i in state:
            self.add(i)

    def __iter__(self):
        for i in self.t:
            if i != self.dela and i != None:
                yield i

    def _hash(self, x):
        return ((self.z * hash(x)) % (1 << w)) >> (w-self.d)

    def find(self, x):
        i = self._hash(x)
        while self.t[i] != None:
            if self.t[i] != self.dela and x == self.t[i]:
                return self.t[i]
            i = (i+1) % len(self.t)

    def add(self, x):

        if self.find(x) != None:
            return False
        if type(x) == tuple:
            x = [x]
        for j in x:
            if 2*(self.q+1) > len(self.t):
                self.resize()
            i = self._hash(j)
            while self.t[i] != None and self.t[i] != self.dela:
                i = (i+1) % len(self.t)
            if self.t[i] == None:
                self.q = self.q+1
            self.n = self.n+1
            self.t[i] = j
        return True

    def discard(self, x):
        if type(x) == tuple:
            x = [x]
        for j in x:
            i = self._hash(j)
            while self.t[i] != None:
                y = self.t[i]
                if y != self.dela and j == y:
                    self.t[i] = self.dela
                    self.n = self.n-1
                    if 8*self.n < len(self.t):
                        self.resize()
                    return y
                i = (i+1) % len(self.t)
        return None

    def resize(self):
        self.d = int(math.log2(3*self.n))
        self.d += 1
        told = self.t
        self.t = [None]*2**self.d
        self.q = self.n
        for x in told:
            if x != None and x != self.dela:
                i = self._hash(x)
                while self.t[i] != None:
                    i = (i+1) % len(self.t)
                self.t[i] = x

    def clear(self):
        self.d = 1
        self.t = [None]*2**self.d
        self.n = 0
        self.q = 0
        self.z = random.randrange(1, 99, 2)


class LinearDict:
    def __init__(self) -> None:
        self.dela = ""
        self.d = 1
        self.t = [None]*2**self.d
        self.q = 0
        self.n = 0
        self.z = random.randrange(1, 99, 2)

    def __setitem__(self, key, value):
        if 2*(self.q+1) > len(self.t):
            self.resize()
        i = self._hash(key)
        while self.t[i] != None and self.t[i] != self.dela and key != self.t[i][0]:
            i = (i+1) % len(self.t)
        if self.t[i] == None:
            self.q = self.q+1
            self.n = self.n+1
        self.t[i] = (key, value)
        return True

    def find(self, x):
        i = self._hash(x)
        while self.t[i] != None:
            if self.t[i][0] == x:
                return i
            i = (i+1) % len(self.t)
        return None

    def __iter__(self):
        for i in self.t:
            if i != self.dela and i != None:
                yield i

    def clear(self):
        self.d = 1
        self.t = [None]*2**self.d
        self.n = 0
        self.q = 0
        self.z = random.randrange(1, 99, 2)

    def _hash(self, x):
        return ((self.z * hash(x)) % (1 << w)) >> (w-self.d)

    def get(self, x, defaultValue):
        i = self._hash(x)
        while self.t[i] != None:
            if x == self.t[i][0]:
                return self.t[i][1]
            i = (i+1) % len(self.t)
        return defaultValue

    def items(self):
        for i in self.t:
            if i != None:
                yield i

    def resize(self):
        self.d = int(math.log2(3*self.n))
        self.d += 1
        told = self.t
        self.t = [None]*2**self.d
        self.q = self.n
        for x in told:
            if x != None and x != self.dela:
                i = self._hash(x[0])
                while self.t[i] != None:
                    i = (i+1) % len(self.t)
                self.t[i] = x


class ChainedSet:

    def __init__(self, state=[]):
        self.d = 1
        self.t = [[] for _ in range(1 << self.d)]
        self.z = random.randrange(1, 99, 2)
        self.n = 0
        for i in state:
            self.add(i)

    def _hash(self, x):
        return ((self.z * hash(x)) % (1 << w)) >> (w-self.d)

    def __iter__(self):
        for ell in self.t:
            for x in ell:
                yield x

    def add(self, x):
        if self.find(x) is not None:
            return False
        if self.n+1 > len(self.t):
            self._resize()
        self.t[self._hash(x)].append(x)
        self.n += 1
        return True

    def _resize(self):
        self.d = 1
        while (1 << self.d) <= self.n:
            self.d += 1
        self.n = 0
        old_t = self.t
        self.t = [[] for _ in range(1 << self.d)]
        for i in range(len(old_t)):
            for x in old_t[i]:
                self.add(x)

    def find(self, x):
        for y in self.t[self._hash(x)]:
            if y == x:
                return y
        return None

    def discard(self, x):
        hashed = self.t[self._hash(x)]
        for y in range(len(hashed)):
            if hashed[y] == x:
                hashed.pop(y)
                self.n -= 1
                if 3*self.n < len(self.t):
                    self._resize()
                return y
        return None


class ChainedDict:
    def __init__(self):
        self.size = 1
        self.map = [[] for _ in range(2**self.size)]
        self.n = 0
        self.z = random.randrange(1, 99, 2)

    def _get_hash(self, key):
        return ((self.z * hash(key)) % (1 << w)) >> (w-self.size)

    def find(self, key):
        for y in self.map[self._get_hash(key)]:
            if y[0] == key:
                return y[1]
        return None

    def add(self, key, value):
        if self.find(key) is not None:
            return False
        if self.n+1 > len(self.map):
            self._resize()
        self.map[self._get_hash(key)].append((key, value))
        self.n += 1
        return True

    def get(self, key, num):
        value = self.find(key)
        if value == None:
            return num
        else:
            return value

    def __iter__(self):
        for i in self.map:
            for x in i:
                yield x

    def _resize(self):
        self.size = 1
        while (1 << self.size) <= self.n:
            self.size += 1
        self.n = 0
        old_t = self.map
        self.map = [[] for _ in range(1 << self.size)]
        for i in range(len(old_t)):
            for x in old_t[i]:
                self.add(x[0], x[1])

    def clear(self):
        self.size = 1
        self.map = [[]]*(1 << self.size)
        self.n = 0

    def __setitem__(self, key, value):
        if self.find(key) is not None:
            temp = self.map[self._get_hash(key)]
            for i in range(len(temp)):
                if temp[i][0] == key:
                    self.map[self._get_hash(key)][i] = (key, value)
        else:
            self.add(key, value)

    def items(self):
        return self.__iter__()


class Life:
    """Life class.

    The state of the game.
    """

    def __init__(self, state: [(int, int)], chain: bool = True) -> None:
        """Initializes game state and internal variables.

        Args:
        - self: automatic object reference.
        - state: initial congifuration - (x,y) coordinates of live cells
        - chain: controls whether to use chaining (True) or linear probiing (False)

        Returns:
        none
        """
        # USet implementations.
        self._alive = None  # intial config: (x, y) coordinates of alive cells.
        self._nbr_count = None  # stores count of live neighbors for cells.
        if chain:
            self._alive: ChainedSet = ChainedSet(state)
            self._nbr_count: ChainedDict = ChainedDict()
        else:
            self._alive: LinearSet = LinearSet(state)
            self._nbr_count: LinearDict = LinearDict()
            # self._nbr_count: LinearDict = {}

    def step(self) -> None:
        """One iteration of the game.

        Applies game rules on current live cells in order to compute the next state of the game.

        Args:
        - self: automatic object reference.

        Returns:
        none
        """
        # Compute neighbors of current live cells.
        deltas = [(-1, -1), (0, -1), (1, -1),
                  (-1,  0),          (1,  0),
                  (-1,  1), (0,  1), (1,  1)]
        neighbors = [(x+dx, y+dy) for x, y in self._alive
                     for dx, dy in deltas]
        # Collect the number of times each coordinate appears as a
        # neighbor. That provides a count of the number of live neighbors of
        # these cells.
        for coord in neighbors:
            self._nbr_count[coord] = self._nbr_count.get(coord, 0) + 1
        # Apply rules based on numberof neighbors.
        for coord, count in self._nbr_count.items():
            # Alive cells with too few or too many alive neighbors die.
            if count == 1 or count > 3:
                self._alive.discard(coord)
            # Cells with 3 alive neighbors come alive.
            elif count == 3:
                self._alive.add(coord)
            # All other live cells survive.
        # Clear for next iteration.
        self._nbr_count.clear()

    def state(self) -> [(int, int)]:
        """Returns the current state of the game.

        Args:
        - self: automatic object reference.

        Returns:
        Coordinates of live cells .
        """
        # self._alive must be iterable, https://stackoverflow.com/a/37639615/1382487
        return list(self._alive)


class Game:
    def run(self, life, config) -> None:
        """Runs the game as per config.

        Args:
        - life: the instance to run.
        - config: contains game configurations.

        Returns:
        nothing.
        """
        # Set up animation if required.
        if config.animate:
            # Use tkinter. Set up the rendering window.
            tk = Tk()
            canvas = Canvas(tk, width=config.width, height=config.height)
            tk.title("Game of Life")
            canvas.configure(background=config.bg_color)
            # Indicate that rendering will be in cells.
            canvas.pack()
            # Number of rendered cells in each direction.
            cells_x = config.width // config.cell_size
            cells_y = config.height // config.cell_size

        # Make the required number of iterations.
        for i in range(config.rounds):
            # Animate if specified.
            if config.animate:
                # Clear canvas and add cells as per current state.
                canvas.delete('all')
                for x, y in life.state():
                    # Wrap cell around screen boundaries. Comment for no wrap.
                    x %= cells_x
                    y %= cells_y
                    # Add cell to canvas.
                    x1, y1 = x * config.cell_size, y * config.cell_size
                    x2, y2 = x1 + config.cell_size, y1 + config.cell_size
                    canvas.create_rectangle(
                        x1, y1, x2, y2, fill=config.cell_color)
                # Render cells, pause for next iteration.
                tk.update()
                time.sleep(0.1 / config.speed)
            # Advanmce the game by one step.
            life.step()


def main():
    config = Config()
    config.animate = True
    config.rounds = 1000
    config.start = Config.glider
    config.speed = 5
    life = Life(config.start)
    Game.run(life, config)


if __name__ == '__main__':
    main()
