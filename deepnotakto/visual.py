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
        self.buttons = {}
        self.init_colors()
        pygame.font.init()
        self.font = pygame.font.SysFont('Calibri Bold', 60)

    def events(self, actions = {pygame.QUIT: sys.exit}):
        """
        Checks and responds to pygame events
        Parameters:
            actions (Dict Pygame Event -> Function) - Actions to take for certain events
        """
        # Loop through events
        for event in pygame.event.get():
            # Find given action to take
            for key in actions.keys():
                if key == event.type:
                    # Call the action
                    actions[key]()

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
            "blue": [0, 0, 255],
            "piece_blue": [29, 135, 229],
            "piece_gray": [144, 164, 174]
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

    @staticmethod
    def norm(x):
        """Normalize an array"""
        xmax, xmin = x.max(), x.min()
        # Catch divide by zero
        if xmax == xmin:
            return x
        return (x - xmin) / (xmax - xmin)

class Game3x3 (Visualization):
    def __init__(self, env, p1, p2, training = False):
        """Initalizes a game on an environment between two players"""
        # Call the parent initializer with the desired screen size
        super(Game3x3, self).__init__([900, 500])
        self.env = env
        self.training = training
        self.p1 = p1
        self.p2 = p2
        self.p1_human = p1.name == "Human"
        self.p2_human = p2.name == "Human"
        self.buttons = {
            "next": (800, 0, 100, 100),
            "0 0": (500, 100, 100, 100),
            "0 1": (600, 100, 100, 100),
            "0 2": (700, 100, 100, 100),
            "1 0": (500, 200, 100, 100),
            "1 1": (600, 200, 100, 100),
            "1 2": (700, 200, 100, 100),
            "2 0": (500, 300, 100, 100),
            "2 1": (600, 300, 100, 100),
            "2 2": (700, 300, 100, 100)
        }
        self.run()

    def display(self, qmatrix, board, banner = "", next = False):
        """Updates the canvas to the desired display screen"""
        self.canvas.fill(self.colors["white"])
        # Write banner to the top center
        if banner != "":
            text_surface = self.font.render(banner, True, self.colors["black"])
            text_rect = text_surface.get_rect()
            self.canvas.blit(text_surface, (self.width // 2 - text_rect[2] // 2,
                                            50 - text_rect[3] // 2))
        # Draw next arrow
        if next:
            pygame.draw.polygon(self.canvas, self.colors["black"],
                                [(825, 25), (825, 75), (875, 50)])
        # DISPLAY CONFIDENCS
        normed = Visualization.norm(qmatrix)
        conf_rects = self.array_to_rect(normed,
                                        [100, 100], 100, self.q_colorfunc)
        self.draw_rects(conf_rects, 10, self.colors["white"])
        # DISPLAY BOARD
        board_rects = self.array_to_rect(board, [500, 100],
                                       100, self.board_colorfunc)
        self.draw_rects(board_rects, 10, self.colors["white"])

    def board_colorfunc(self, x):
        """Color function for a regular board"""
        if int(round(x)) == 1:
            return self.colors["piece_blue"]
        else:
            return self.colors["piece_gray"]

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
                qs = np.reshape(self.p1.get_Q(self.env.observe()), [3, 3])
                self.p1.act(self.env, training=False)
                self.env.turn += 1
                banner = "PLAYER 2"
            else:
                # Banner
                banner = "PLAYER 1"
                qs = np.zeros([3, 3])
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
                if button == "next":
                    button = ""
                    # Update banner
                    if banner == "Player attempted illegal move":
                        banner == "PLAYER {}".format(self.env.turn % 2)
                    # Copy the board for later comparison pre and post move
                    b_copy = copy(self.env.board)
                    # Play the agent corresponding to the current turn
                    if self.env.turn % 2 == 0:
                        qs = np.reshape(self.p1.get_Q(b_copy), [3, 3])
                        self.p1.act(self.env, training = self.training)
                    else:
                        qs = np.reshape(self.p2.get_Q(b_copy), [3, 3])
                        self.p2.act(self.env, training = self.training)
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
    e = Env(3)
    p = Q([9, 50, 100, 9], "agents/params/Q[9, 50, 100, 9]_WORKING_WITH_MIDDLE.npz")
    h = Human()
    Game3x3(e, p, h)