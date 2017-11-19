# visual.py
# Abraham Oliver, 2017
# Deep Notakto Project

import sys, pygame
import numpy as np
from matplotlib.pyplot import get_cmap
from copy import copy
from tensorflow.python.framework.errors_impl import InvalidArgumentError
import util

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
            colorfunc (Float -> Float[3]) - Function to seconds_to_time a number to a color
        Returns:
            List of N * M colored rectangles [x][y][width][height][color]
        """
        rects = []
        # Loop through rows
        for n in range(a.shape[0]):
            # Loop over the columns
            for m in range(a.shape[1]):
                # Get current entry
                x = a[m, n]
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

class GameWithConfidences (Visualization):
    def __init__(self, env, a1, a2, max_games = -1, piece_size = 100, show_confidences = True):
        """
        Initalizes a GUI game on an environment between two players
        Parameters:
            env (Environment) - Environment to play a game on
            a1 (Agent) - Agent 1
            a2 (Agent) - Agent 2
            max_games (int) - Number of games to play (negative = until close)
            piece_size (int) - Side lenghth of a piece
            show_confidences (bool) - Show the agent confidence matrix or not
        """
        # Get the desired screen size (depends on piece_size and board size)
        self.shape = env.shape
        self.side = self.shape[0]
        self.piece_size = piece_size
        self.show_confidences = show_confidences
        if show_confidences:
            # 3 spacing pieces and 2 game board
            width = piece_size * (3 + self.side * 2)
        else:
            # 2 spacing pieces and a game board
            width = piece_size * (2 + self.side)
        # 2 spacing pieces and a game board
        height = piece_size * (2 + self.side)
        # Call parent intiializer
        super(GameWithConfidences, self).__init__([width, height])
        self.env = env
        self.max_games = max_games
        self.a1 = a1
        self.a2 = a2
        self.a1_human = a1.name == "Human"
        self.a2_human = a2.name == "Human"
        self.define_buttons()
        self.colors["piece_closed"] = [29, 135, 229]
        self.colors["piece_open"] = [144, 164, 174]
        if (self.width // 30) < 25:
            self.font = pygame.font.SysFont('Calibri Bold', 25)
        else:
            self.font = pygame.font.SysFont('Calibri Bold', self.width // 30)
        # Run game (and end if invalid values inhibit agents)
        self.run()
        """
        try:
            self.run()
        except InvalidArgumentError as error:
            print("Tensor has NaN or Inf values")
            pygame.quit()
        except:
            pygame.quit()"""

    def define_buttons(self):
        """Define the buttons coordinates for the given board size"""
        buttons = {
            "next": (self.width - self.piece_size, 0, self.piece_size, self.piece_size)
        }
        start = [self.piece_size, self.piece_size]
        for n in range(self.side):
            for m in range(self.side):
                buttons["{0} {1}".format(m, n)] = (
                    start[0] + n * self.piece_size,
                    start[1] + m * self.piece_size,
                    self.piece_size, self.piece_size
                )
        self.buttons = buttons

    def display(self, board, confidences = None, banner = "", next = False):
        """
        Updates the canvas to the desired display screen
        Parameters:
            board ((N, N) array) - Current board state
            confidences ((N, N) array) - Confidences of the previous agent's move
                                            (only if self.show_confidences)
            banner (string) - Banner to display on the top of the screen
            next (bool) - Show next button or not
        """
        # Fill the canvas (background color)
        self.canvas.fill(self.colors["white"])
        # Write banner to the top center (if there is a banner to write)
        if banner != "":
            # Render onto a text surface
            text_surface = self.font.render(banner, True, self.colors["black"])
            text_rect = text_surface.get_rect()
            # Place the surface centered on the screen
            self.canvas.blit(text_surface, (self.width // 2 - text_rect[2] // 2,
                                            self.piece_size // 2 - text_rect[3] // 2))
        # Draw next arrow (if enabled)
        if next:
            # A triangle in the top right corner
            pygame.draw.polygon(self.canvas, self.colors["black"],
                                [(self.width - ((3 * self.piece_size) // 4),
                                  self.piece_size // 4),
                                 (self.width - ((3 * self.piece_size) // 4),
                                  (3 * self.piece_size) // 4),
                                  (self.width - (self.piece_size // 4),
                                   self.piece_size // 2)])
        # DISPLAY CONFIDENCS
        if self.show_confidences:
            # Normalize the confidences
            normed = util.norm(confidences)
            # Get the colored rectangles representing the confidences
            start_conf_point = [self.piece_size * (self.side + 2),
                                self.piece_size]
            conf_rects = self.array_to_rect(normed, start_conf_point,
                                            self.piece_size, self.q_colorfunc)
            # Draw the rectangles
            self.draw_rects(conf_rects, self.piece_size // 10, self.colors["white"])
        # DISPLAY BOARD
        # Get the colored rectangles representing the game board
        board_rects = self.array_to_rect(board,
                                         [self.piece_size, self.piece_size],
                                         self.piece_size,
                                         self.board_colorfunc)
        # Draw the rectangles
        self.draw_rects(board_rects, self.piece_size // 10 , self.colors["white"])
        # Update the screen
        self.update()

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
        """Run the pygame event loop and return any buttons pressed"""
        # Button pressed (originally none)
        button = ""
        # Check event queue
        for event in pygame.event.get():
            # If exit button clicked, exit game
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # If a click, see if it is over a virtual button
            elif event.type == pygame.MOUSEBUTTONUP:
                button = self.get_button(event.pos)
        # Return the virtual button clicked
        return button

    def run(self):
        """Runs the game"""
        # ---------- GAME SET LOOP ----------
        games = 0
        confidences = None
        self.a1.new_episode()
        self.a2.new_episode()
        while True:
            games += 1
            # Reset environment
            self.env.reset()
            # Is the game loop finished
            done = False
            # If the first player is a computer, have it play
            if not self.a1_human:
                if self.show_confidences:
                    confidences = self.a1.get_Q(self.env.observe())
                self.a1.act(self.env)
                self.env.turn += 1
            # Otherwise setup for a human player's first turn
            elif self.show_confidences:
                confidences = np.zeros(self.shape)
            button = ""
            # ---------- MAIN GAME ----------
            while not done:
                # Run event loop and get a button press if one occurs
                button = self.events()
                # Set the banner for the current player
                banner = "PLAYER {}".format((self.env.turn % 2) + 1)
                # Display the game screen
                board = self.env.observe()
                self.display(board, confidences, banner,
                             next = not (self.a1_human and self.a2_human))
                h_advance = False
                # ---------- HUMAN MOVE GUI LOOP (if needed) ----------
                if (self.env.turn % 2 == 0 and self.a1_human) or \
                    (self.env.turn % 2 == 1 and self.a2_human):
                    # Run until a valid human move has been played
                    while True:
                        # Draw screen
                        self.display(board, confidences, banner, next = False)
                        # Run event loop and fetch button
                        button = self.events()
                        # Parse mouse input (decide which move the human wants)
                        if not (button in ["", "next"]):
                            # Get indicies
                            n, m = [int(i) for i in button.split()]
                            # If the move is invalid, do nothing
                            if board[n, m] != 0:
                                button = ""
                                continue
                            # Make a move vector
                            action = np.zeros(self.env.shape, dtype = np.int32)
                            action[n, m] = 1
                            # Play the move
                            self.env.act(action)
                            # Increase turn counter
                            self.env.turn += 1
                            # Exit human loop
                            break
                    # If human ended game, exit game loop
                    if done:
                        break
                    # Check for ending
                    done = False if self.env.is_over() == 0 else True
                    # If over, end game
                    if done: break
                    # If not over, advance to next move automatically
                    else: h_advance = True
                # ---------- END HUMAN MOVE GUI LOOP ----------
                # ---------- COMPUTER MOVE (if needed) ----------
                # If advanced to next move (by human or by human move) and
                #   next player is an AI, enter AI loop
                if (button == "next" or h_advance) and not (self.a1_human and self.a2_human):
                    # Clear button
                    button = ""
                    h_advance = False
                    # Update banner
                    banner = "PLAYER {}".format((self.env.turn % 2) + 1)
                    # Play the agent corresponding to the current turn
                    player = [self.a1, self.a2][self.env.turn % 2]
                    if self.show_confidences:
                        confidences = player.get_Q(self.env.observe())
                    observation = player.act(self.env)
                    # Update turn counter
                    self.env.turn += 1
                    # Catch illegal move
                    if observation["info"]["illegal"]:
                        banner = "Player attempted illegal move"
                        done = True
                        break
                    # Check for a win
                    done = False if self.env.is_over() == 0 else True
                # ---------- END COMPUTER MOVE ----------
            # ---------- END MAIN GAME ----------
            # End game
            self.a1.save_episode()
            self.a2.save_episode()
            # If individual game is over
            if banner != "Player attempted illegal move":
                banner = "PLAYER {} WINS".format(self.env.is_over())
            # ---------- FINAL SCREEN ----------
            board = self.env.observe()
            while True:
                # Check if next button is permitted
                allow_next = (games < self.max_games) or (self.max_games < 0)
                # Run event loop and fetch button press
                button = self.events()
                # Display the game screen
                self.display(board, confidences, banner, next = allow_next)
                # Continue game set
                if button == "next" and allow_next:
                    break
            # ---------- END FINAL SCREEN ----------
        # ---------- END GAME SET LOOP ----------

if __name__ == "__main__":
    from agents.random_agent import RandomAgent
    from environment import Env
    from agents.human import Human
    from agents.Q import Q
    e = Env(3)
    p2 = Human()
    # p2 = RandomAgent(e)
    p1 = Q([9, 100, 9], gamma = .5, epsilon = .1, name = "AA",
                training = {"type": "episodic", "learn_rate": 1e-1, "rotate": True})
    vis = GameWithConfidences(e, p1, p2, show_confidences = True)