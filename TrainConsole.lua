-- Load required packages
require 'nn'
require 'optim'


-- Initialise environment and state
env = {}

gridSize = 10
fruitRow = 1
fruitColumn = math.random(1, gridSize)
basketPosition = math.random(2, gridSize - 1)
state = torch.Tensor({ fruitRow, fruitColumn, basketPosition })

canvas = torch.Tensor(gridSize, gridSize):zero()
canvas[state[1]][state[2]] = 1 -- Draw the fruit.
-- Draw the basket. The basket takes the adjacent two places to the position of basket.
canvas[gridSize][state[3] - 1] = 1
canvas[gridSize][state[3]] = 1
canvas[gridSize][state[3] + 1] = 1

currentState = canvas:clone()


-- Create network
nbStates = 100
hiddenSize = 100
nbActions = 3

model = nn.Sequential()
model:add(nn.Linear(nbStates, hiddenSize))
model:add(nn.ReLU())
model:add(nn.Linear(hiddenSize, hiddenSize))
model:add(nn.ReLU())
model:add(nn.Linear(hiddenSize, nbActions))

-- epsilon-greedy exploration: select action
--[[ Helper function: Chooses a random value between the two boundaries.]] --
function randf(s, e)
    return (math.random(0, (e - s) * 9999) / 10000) + s;
end

epsilon = 1
if (randf(0, 1) <= epsilon) then
    action = math.random(1, nbActions)
else
    q = model:forward(currentState)
    _, index = torch.max(q, 1)
    action = index[1]
end

-- Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
epsilonMinimumValue = 0.001
if (epsilon > epsilonMinimumValue) then
    epsilon = epsilon * 0.999
end


-- carry action on environment
if (action == 1) then
    action = -1
elseif (action == 2) then
    action = 0
else
    action = 1
end

-- observe new state
fruitRow = fruitRow + 1
 -- The min/max prevents the basket from moving out of the grid.
basketPosition = math.min(math.max(2, basketPosition + action), gridSize - 1)
fruitColumn = fruitColumn -- it's fixed
state = torch.Tensor({ fruitRow, fruitColumn, basketPosition })

-- initialise canvas of next state 
canvas = torch.Tensor(gridSize, gridSize):zero()
canvas[state[1]][state[2]] = 1 -- Draw the fruit.
-- Draw the basket. The basket takes the adjacent two places to the position of basket.
canvas[gridSize][state[3] - 1] = 1
canvas[gridSize][state[3]] = 1
canvas[gridSize][state[3] + 1] = 1

nextState = canvas:clone()

-- observe reward
reward = 0
if (fruitRow == gridSize - 1) then -- If the fruit has reached the bottom.
    if (math.abs(fruitColumn - basketPosition) <= 1) then -- Check if the basket caught the fruit.
         reward = 1
    else
        reward = -1
    end
else
    reward = 0
end

winCount = 0
gameOver = false
if (reward == 1) then winCount = winCount + 1 end
if (nextState[1] == gridSize - 1) then gameOver = true end


-- Initialise replay memory
memory = {}


-- save an experience to replay memory
table.insert(memory, {
    inputState = currentState:view(-1),
    action = action,
    reward = reward,
    nextState = nextState:view(-1),
    gameOver = gameOver
});

-- choose mini-batch of transitions
batchSize = 50
memoryLength = #memory
chosenBatchSize = math.min(batchSize, memoryLength)
inputs = torch.Tensor(chosenBatchSize, nbStates):zero()
targets = torch.Tensor(chosenBatchSize, nbActions):zero()

i = 1
-- Choose a random memory experience to add to the batch.
randomIndex = math.random(1, memoryLength)
memoryInput = memory[randomIndex]

-- Calculate Q-value
if (memoryInput.gameOver) then
    target[memoryInput.action] = memoryInput.reward
else
   discount = 0.9  -- discount factor
   
   -- Gives us Q_sa for all actions
   target = model:forward(memoryInput.inputState)
   
   -- reward + discount(gamma) * max_a' Q(s',a')
   -- We are setting the Q-value for the action to  r + γmax a’ Q(s’, a’). 
   -- The rest stay the same to give an error of 0 for those outputs.
   -- the max q for the next state.
   nextStateMaxQ = torch.max(model:forward(memoryInput.nextState), 1)[1]
   target[memoryInput.action] = memoryInput.reward + discount * nextStateMaxQ
end
-- Update the inputs and targets.
inputs[i] = memoryInput.inputState
targets[i] = target

-- Train the network which returns the error.
criterion = nn.MSECriterion()

sgdParams = {
    learningRate = 0.1,
    learningRateDecay = 1e-9,
    weightDecay = weightDecay,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}

loss = 0
x, gradParameters = model:getParameters()
function feval(x_new)
    gradParameters:zero()
    local predictions = model:forward(inputs)
    local loss = criterion:forward(predictions, targets)
    local gradOutput = criterion:backward(predictions, targets)
    model:backward(inputs, gradOutput)
    return loss, gradParameters
end

_, fs = optim.sgd(feval, x, sgdParams)
loss = loss + fs[1]
