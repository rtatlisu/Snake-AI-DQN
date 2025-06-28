# Snake-AI-DQN
## Introduction
I went for RL with Deep Q-Learning for this project to learn how this method exactly works.
My expections for this project were not to make the snake win the game, since - atleast to the best of my knowledge - this would be rather feasibile with a CNN.
In the following sections I'll explain the project and also why exactly I think that DQN isn't ideally to make the 'perfect' snake AI.

## Approach
I chose a dense structure with 11 inputs, 64 hidde nodes an 3 output nodes.
The input nodes, i.e. the state, consist of only binary variables. These are direction (up, right, down, left), fruit direction (up, right, down, left) and dangers (straight, left, right).
With this info the snake is able to mostly make the right decisions.
The snake is rewarded for eating a fruit and punished for moving out of bounds or into itself. Additionally, it has a lifespan (limited move count) that correlates to its length and this limit is recalculated when it eats a fruit.

## Visuals

