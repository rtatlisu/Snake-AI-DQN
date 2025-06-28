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

The values for the hyperparameters are: 
* learning rate = 0.001
* epsilon = 1
* replay buffer size = 10000
* batch size = up to 1000
* optimizer = adam
* target network update rate = every 100 moves
* gamma = 0.9
* max move count = min(100 + 2 * len(self.snake_segments), 200)
* activation function = RelU

## Visuals
Below you can see the peformance of the snake after almost 2000 iterations.
I tweaked with epsilon quite a bit, because it seemed to bottleneck the average performance a lot. I settled at a starting epsilon of 1 which is then reduced by 3% per action taken. The lowest the epsilon can be is 0.005.
As seen in the gif, the major problem of the snake is that it runs into itself, because it takes a route where it can't avoid a collision with itself anymore.

![me](https://github.com/rtatlisu/Snake-AI-DQN/blob/main/snake_preview.gif)

## Conclusion
I'm quite happy with the result, as I learned a lot about DQN and Torch.
The issue with the approach I chose is that the snake can never learn to avoid running into a dead end that is created by its own body.
For that reason I didn't even start fiddling around with other hyperparameters apart from epsilon since it wouldn't solve my problem.
With the settings described above, the snake reaches an average score of just under 30, which isn't too bad.
A more optimal approach would be to give the snake info on the entire playing field and/or to include pathfinding algorithms like A*, but since my goals was primarily to learn about DQN and some libraries, I left it at that state.


