#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: visual.py                                                    #
#  Abraham Oliver, 2018                                               #
#######################################################################
import pygame

# Initialize Pygame
pygame.init()


class Visualization (object):
    def __init__(self, size):
        """
        Initializes a visualization with a given size

        Args:
            size: (int[2]) Width-by-height tuple for visualization size
        """
        self.size = self.width, self.height = size
        self.canvas = pygame.display.set_mode(self.size)
        self.define_buttons()
        self.init_colors()
        pygame.font.init()

    def update(self):
        """ Update display """
        pygame.display.flip()

    def array_to_rect(self, a, start_coords, block_size, colorfunc):
        """
        Converts an array into a list of colored and located rectangles
        Args:
            a: (array) Array to be converted
            start_coords: (int[2]) Location of first corner
            block_size: (int) Size of each block
            colorfunc: (Float -> Float[3]) Function from a value to a color
        Returns:
            List of colored rectangles: (x, y, width, height, color)
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

    def draw_rects(self, rects, border_size = 0, border_color = [0, 0, 0]):
        """
        Draw a list of rectangles to the canvas

        Args:
            rects: ((x, y, w, h, color)[]) List of rectangles to draw
            border_size: (int) Width of border
            border_color: (int[3]) Color tuple for color of border
        """
        for r in rects:
            pygame.draw.rect(self.canvas, r[4], r[:4])
            # Draw border if requested
            if border_size > 0:
                pygame.draw.rect(self.canvas, border_color, r[:4], border_size)

    def init_colors(self):
        """ Initializes color dictionary """
        self.colors = {
            "white": [255, 255, 255],
            "black": [0, 0, 0],
            "red": [255, 0, 0],
            "green": [0, 255, 0],
            "blue": [0, 0, 255]
        }

    def get_button(self, cursor_loc, buttons = None):
        """
        Returns the on-screen button that the user has clicked

        Args:
            cursor_loc: (int[2]) Location of the cursor
            buttons: (dict)  Dictionary of button names to their rectangle areas
        Returns:
            string - Name of button (default "")
        Note:
            Includes right and bottom sides
            Does not include top and left sides
        """
        if buttons is None:
            buttons = self.buttons
        x, y = cursor_loc
        for name in buttons:
            bx, by, bw, bh = buttons[name]
            if x > bx and x <= (bx + bw) and y > by and y <= (by + bh):
                return name
        # Default
        return ""

    def define_buttons(self):
        """ Define clickle button locations """
        self.buttons = {}