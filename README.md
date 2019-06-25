# NeuroEvolution-FlappyBird
## Using Neuroevolution to play FlappyBird.

Using my own [NeuralNetwork code](https://github.com/ta9ay/NeuralNet-from-scratch "NeuralNet-from_scratch") to write a NeuroEvolutionary algorithm to play [FlappyBird](https://github.com/jmathison/Flappython)
#### Possible Improvements:
1. The scoring mechanism to record each bird's score might be improved. Training times vary based on how much the lift of the bird is when it jumps. On a lift on -15, scoring 100+ is possible within the first 4-5 generations. On a lift of -18, it takes much longer to score 100+ points. Around 70-80 generations.
2. Saving the best bird and then having the option to load that bird.
![screen: ](https://github.com/ta9ay/NeuroEvolution-FlappyBird/blob/master/Screenshot.png)
### Requirements
1. PyGame
2. NumPy
