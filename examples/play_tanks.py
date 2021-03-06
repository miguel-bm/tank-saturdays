#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import gym
from gym import wrappers
import tank_saturdays
from pathlib import Path
import termios, fcntl, sys, os
import time
import numpy as np


def play_tanks():
    typer.echo(f"Playing tank-saturdays with a human player.")

    # Make and seed the environment
    env = gym.make('TankSaturdays-v0')
    rendered=env.render(mode='console')
    obs = env.reset()

    pressed_keys = []
    transpose = True
    running = True
    env_done = False

    turn = 0
    total_reward = 0
    while running:
        env.render("console")

        # Process events
        fd = sys.stdin.fileno()

        oldterm = termios.tcgetattr(fd)
        newattr = termios.tcgetattr(fd)
        newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, newattr)

        oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

        if env_done:  # Natural end of a simulation, game is resetted
            typer.echo(f"Game reached end-state in turn {turn}.")
            if reward == 0:
                typer.echo("It was a draw")
            elif reward == 1:
                typer.echo("Black tank won")
            elif reward == -1:
                typer.echo("White tank won")
            running = False
            break

        try:
            while 1:
                try:
                    c = sys.stdin.read(1)
                    if c:
                        if str(c) == "A":
                            action = 4
                            break
                        elif str(c) == "B":
                            action = 2
                            break
                        elif str(c) == "C":
                            action = 3
                            break
                        elif str(c) == "D":
                            action = 1
                            break
                        elif str(c) == "e":
                            action = 0
                            break
                        elif str(c) == "w":
                            action = 8
                            break
                        elif str(c) == "a":
                            action = 5
                            break
                        elif str(c) == "s":
                            action = 6
                            break
                        elif str(c) == "d":
                            action = 7
                            break
                        elif str(c) == "q":
                            typer.echo(f"Quit game.")
                            running = False
                            break
                except IOError: pass
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
            fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
        if running == False:
            break

        time.sleep(0.01)

        prev_obs = obs
        observation, reward, env_done, _ = env.step(
            action, np.random.randint(9))
        turn += 1


if __name__ == "__main__":
    typer.run(play_tanks)
