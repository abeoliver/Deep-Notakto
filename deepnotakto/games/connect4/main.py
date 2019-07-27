from deepnotakto.games.connect4.game import RandomAgent, Human, play
from os import system


def main():
    r = RandomAgent()
    h = Human()
    play(h,r, -1, clear_func = lambda: system('clear'))


if __name__ == "__main__":
    main()
