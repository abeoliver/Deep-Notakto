# visual.py
# Abraham Oliver, 2017
# Deep Notakto Project

import sys, pygame
import numpy as np
from matplotlib.pyplot import get_cmap
from copy import copy

# Initialize Pygame
pygame.init()

class Visualization (object):
    def __init__(self, size):
        """Initializes a visualization"""
        self.size = self.width, self.height = size
        self.canvas = pygame.display.set_mode(self.size)
        self.define_buttons()
        self.init_colors()
        pygame.font.init()

    def update(self):
        """Update display"""
        pygame.display.flip()

    def array_to_rect(self, a, start_coords, block_size, colorfunc):
        """
        Converts an array into a list of colored and located rectangles
        Parameters:
            a ((N, M) array) - Array to be converted
            start_coords ([Int, Int]) - Location of first corner
            block_size (int) - Size of each block
            colorfunc (Float -> Float[3]) - Function to convert a number to a color
        Returns:
            List of N * M colored rectangles [x][y][width][height][color]
        """
        rects = []
        # Loop through rows
        for n in range(a.shape[0]):
            # Loop over the columns
            for m in range(a.shape[1]):
                # Get current entry
                x = a[n, m]
                # Get color for item
                color = colorfunc(x)
                # Get located coordinates
                loc = [start_coords[0] + n * block_size,
                       start_coords[1] + m * block_size]
                # Add rectangle
                rects.append((loc[0], loc[1], block_size, block_size, color))
        # Return list
        return rects

    def draw_rects(self, rects, border_size = 0, color = [0, 0, 0]):
        """Draw a list of rectangles to the canvas"""
        for r in rects:
            pygame.draw.rect(self.canvas, r[4], r[:4])
            # Draw border if requested
            if border_size > 0:
                pygame.draw.rect(self.canvas, color, r[:4], border_size)

    def init_colors(self):
        """Initializes color dictionary"""
        self.colors = {
            "white": [255, 255, 255],
            "black": [0, 0, 0],
            "red": [255, 0, 0],
            "green": [0, 255, 0],
            "blue": [0, 0, 255]
        }

    def get_button(self, cursor_loc):
        """
        Returns the on-screen button that the user has clicked
        Parameters:
            cursor_loc ([int, int]) - Location of the cursor
            buttons (Dict String ->) - Dictionary of button names to their rectangles
        Returns:
            string - Name of button (default "")
        Note:
            Includes right and bottom sides
            Does not include top and left sides
        """
        x, y = cursor_loc
        for name in self.buttons:
            bx, by, bw, bh = self.buttons[name]
            if x > bx and x <= (bx + bw) and y > by and y <= (by + bh):
                return name
        # Default
        return ""

    def define_buttons(self):
        self.buttons = {}

    @staticmethod
    def norm(x):
        """Normalize an array"""
        xmax, xmin = x.max(), x.min()
        # Catch divide by zero
        if xmax == xmin:
            return x
        return (x - xmin) / (xmax - xmin)

class GameWithConfidences (Visualization):
    def __init__(self, env, p1, p2, training = False, piece_size =  100, learn_rate = .0001):
        """Initalizes a game on an environment between two players"""
        # Call the parent initializer with the desired screen size
        self.shape = env.shape
        self.side = self.shape[0]
        self.piece_size = piece_size
        width = piece_size * (3 + self.side * 2)
        height = piece_size * (2 + self.side)
        super(GameWithConfidences, self).__init__([width, height])
        self.env = env
        self.training = training
        self.learn_rate = learn_rate
        self.p1 = p1
        self.p2 = p2
        self.p1_human = p1.name == "Human"
        self.p2_human = p2.name == "Human"
        self.define_buttons()
        self.colors["piece_closed"] = [29, 135, 229]
        self.colors["piece_open"] = [144, 164, 174]
        self.font = pygame.font.SysFont('Calibri Bold', self.width // 30)
        # Run game
        try:
            self.run()
        except:
            pygame.quit()

    def define_buttons(self):
        """Define the buttons for the given board size"""
        buttons = {
            "next": (self.width - self.piece_size, 0, self.piece_size, self.piece_size)
        }
        start = [(2 + self.side) * self.piece_size, self.piece_size]
        for n in range(self.side):
            for m in range(self.side):
                buttons["{0} {1}".format(m, n)] = (
                    start[0] + n * self.piece_size,
                    start[1] + m * self.piece_size,
                    self.piece_size, self.piece_size
                )
        self.buttons = buttons

    def display(self, qmatrix, board, banner = "", next = False):
        """Updates the canvas to the desired display screen"""
        self.canvas.fill(self.colors["white"])
        # Write banner to the top center
        if banner != "":
            text_surface = self.font.render(banner, True, self.colors["black"])
            text_rect = text_surface.get_rect()
            self.canvas.blit(text_surface, (self.width // 2 - text_rect[2] // 2,
                                            self.piece_size // 2 - text_rect[3] // 2))
        # Draw next arrow
        if next:
            pygame.draw.polygon(self.canvas, self.colors["black"],
                                [(self.width - ((3 * self.piece_size) // 4),
                                  self.piece_size // 4),
                                 (self.width - ((3 * self.piece_size) // 4),
                                  (3 * self.piece_size) // 4),
                                  (self.width - (self.piece_size // 4),
                                   self.piece_size // 2)])
        # DISPLAY CONFIDENCS
        normed = Visualization.norm(qmatrix)
        conf_rects = self.array_to_rect(normed,
                                        [self.piece_size, self.piece_size],
                                        self.piece_size, self.q_colorfunc)
        self.draw_rects(conf_rects, self.piece_size // 10, self.colors["white"])
        # DISPLAY BOARD
        board_rects = self.array_to_rect(board,
                                         [(self.side + 2) * self.piece_size,
                                          self.piece_size],
                                         self.piece_size,
                                         self.board_colorfunc)
        self.draw_rects(board_rects, self.piece_size // 10 , self.colors["white"])

    def board_colorfunc(self, x):
        """Color function for a regular board"""
        if int(round(x)) == 1:
            return self.colors["piece_closed"]
        else:
            return self.colors["piece_open"]

    def q_colorfunc(self, x, cmap = "viridis"):
        """
        Colorfunction for Q values
        Parameters:
            x ((N, N) array) - Q value matrix that is normalized to [0, 1]
            cmap (string) - Matplotlib colormap name
        """
        return np.int32(np.multiply(255, get_cmap(cmap)(x)))

    def events(self):
        """Runs the event loop and returns any buttons pressed"""
        button = ""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONUP:
                button = self.get_button(event.pos)
        return button

    def run(self):
        """Runs the game between the two agents"""
        # Play games until exited
        while True:
            # Reset environment
            self.env.reset()
            # Is the game loop finished
            done = False
            # If the first player is a computer, have it play
            if not self.p1_human:
                qs = np.reshape(self.p1.get_Q(self.env.observe()), self.shape)
                self.p1.act(self.env, training = self.training, learn_rate = self.learn_rate)
                self.env.turn += 1
                banner = "PLAYER 2"
            else:
                # Banner
                banner = "PLAYER 1"
                qs = np.zeros(self.shape)
            board = self.env.observe()
            button = ""
            # Play game
            while not done:
                # Run event loop, drawing, and update
                button = self.events()
                self.display(qs, board, banner, next = True)
                self.update()
                # Run human turn if turn
                if (self.env.turn % 2 == 0 and self.p1_human) or \
                    (self.env.turn % 2 == 1 and self.p2_human):
                    # Human turn
                    while True:
                        # Run event loop, drawing, and update
                        button = self.events()
                        self.display(qs, board, banner, next = True)
                        self.update()
                        # Parse mouse input
                        if not (button in ["", "next"]):
                            # Get indicies
                            n, m = [int(i) for i in button.split()]
                            # Make a move
                            move = np.zeros(self.env.shape, dtype = np.int32)
                            move[m, n] = 1
                            # Play the move
                            self.env.act(move)
                            self.env.turn += 1
                            break
                    # Continue game loop
                    board = self.env.observe()
                    done = False if self.env.is_over() == 0 else True
                    if done: break
                    else: button = "next"
                # On user command, run next agent turn
                if button == "next" and not (self.p1_human and self.p2_human):
                    button = ""
                    # Update banner
                    if banner == "Player attempted illegal move":
                        banner == "PLAYER {}".format(self.env.turn % 2)
                    # Copy the board for later comparison pre and post move
                    b_copy = copy(self.env.board)
                    # Play the agent corresponding to the current turn
                    if self.env.turn % 2 == 0:
                        qs = np.reshape(self.p1.get_Q(b_copy), self.shape)
                        self.p1.act(self.env, training = self.training,
                                    learn_rate = self.learn_rate)
                    else:
                        qs = np.reshape(self.p2.get_Q(b_copy), self.shape)
                        self.p2.act(self.env, training = self.training,
                                    learn_rate = self.learn_rate)
                    board = self.env.observe()
                    # Change turn
                    self.env.turn += 1
                    # Catch double illegal moves
                    if np.equal(b_copy, self.env.board).all() and self.env.turn != 0:
                        banner = "Player attempted illegal move"
                        done = True
                        break
                    done = False if self.env.is_over() == 0 else True
            if banner != "Player attempted illegal move":
                banner = "PLAYER {} WINS".format(self.env.is_over())
            # Leave on final screen until clicked forward
            while True:
                button = self.events()
                self.display(qs, board, banner, next = True)
                self.update()
                if button == "next":
                    break

if __name__ == "__main__":
    from environment import Env
    from agents.Q import Q
    from agents.human import Human
    from agents.random_agent import RandomAgent
    from agents.random_plus import RandomAgentPlus
    size = 4
    e = Env(size)
    p = Q([size * size, size * size])
    # p = Human()
    h = Human()
    GameWithConfidences(e, p, h, piece_size = 150)