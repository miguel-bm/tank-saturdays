# Tank Saturdays

This repository contains a [gym](https://github.com/openai/gym) environment that will be used in a competition between the members of the Reinforcement Learning track form the 3rd Edition of AI Saturdays Madrid.

The game consists of a simplified battle between two tanks than move over a grid. Each of the tanks has limited fuel and ammo, and must try to defeat the opponent by shooting him three times. Both movement and shooting are restricted to the four cardinal directions, and bullets travel at a limited speed. 

Each turn in the game is played out at the same time for both players. Each tank does not know what the opponent will do that turn. The nine possible choices are to move in one direction, to shoot in one direction, or to stay idle.

The battlefield is obstructed by some randomly placed walls, which the tanks can neither break nor move through. The tanks can't exit the bounds of the battlefield.

This game was created and developed by [Pablo Talavante](https://github.com/pablotalavante) and [Miguel Blanco](https://github.com/miguel-bm).
