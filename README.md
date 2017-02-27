# QlearningExample.torch
![TorchPlaysCatch](https://github.com/SeanNaren/TorchQLearningExample/raw/master/images/torchplayscatch.gif)

Torch plays catch! Based on Eder Santanas' [implementation](https://gist.github.com/EderSantana/c7222daa328f0e885093) in keras. Highly recommend reading his informative and easy to follow blog post [here](https://edersantana.github.io/articles/keras_rl/).

Agent has to catch the fruit before it falls to the ground. Agent wins if he succeeds to catch the fruit, loses if he fails.

## Dependencies

To install torch7 follow the guide <a href="http://torch.ch/docs/getting-started.html">here</a>.

Other dependencies can be installed via luarocks:

```
luarocks install optim
luarocks install image
```

## How to run

To train a model, run the `Train.lua` script. You can configure parameters such as below:

```
th Train.lua -epoch 1000 #Configures the number of epochs. More parameters are available, check scrip.t
```

## Visualization

To visualise the agent playing the game after training a model, use the `TorchPlaysCatch.lua` script using qlua rather than th as below:

```
qlua TorchPlaysCatch.lua
```

Much like the train script, there are configurable options. Check the script for more details!

## Acknowledgements
Eder Santana, Keras plays catch, (2016), GitHub repository, https://gist.github.com/EderSantana/c7222daa328f0e885093



Guest Post (Part I): Demystifying Deep Reinforcement Learning
https://www.nervanasys.com/demystifying-deep-reinforcement-learning/

	- What are the main challenges in reinforcement learning? 
		- credit assignment problem
		- exploration-exploitation dilemma
	- How to formalize reinforcement learning in mathematical terms? 
		- Markov Decision Process
	- How do we form long-term strategies?
		- discounted future reward
	- How can we estimate or approximate the future reward?
		- Simple table-based Q-learning algorithm
	- What if our state space is too big? 
		- Deep Q Network
	- What do we need to make it actually work? 
		- Experience replay : stabilizes the learning with neural networks.
	- Are we done yet? Finally we will consider some simple solutions to the exploration-exploitation problem.


Keras plays catch, a single file Reinforcement Learning example
https://edersantana.github.io/articles/keras_rl/



SeanNaren/QlearningExample.torch
https://github.com/SeanNaren/QlearningExample.torch






# todo
https://www.slideshare.net/carpedm20/reinforcement-learning-an-introduction-64037079?qid=23c83803-34c2-4bf3-a33e-e702909c50c8&v=&b=&from_search=1

http://keunwoochoi.blogspot.kr/2016/06/andrej-karpathy.html

http://karpathy.github.io/2016/05/31/rl/


http://www.modulabs.co.kr/RL_library/2136
