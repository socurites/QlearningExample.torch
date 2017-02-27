require 'nn'
require 'CatchEnvironment'
require 'optim'

epsilon = 1
epsilonMinimumValue = 0.001
nbActions = 3
epoch = 1000
hiddenSize = 100
maxMemory = 500
batchSize = 50
gridSize = 10
nbStates = 100
discount = 0.9

math.randomseed(os.time())

--[[ Helper function: Chooses a random value between the two boundaries.]] --
function randf(s, e)
    return (math.random(0, (e - s) * 9999) / 10000) + s;
end

--[[ The memory: Handles the internal memory that we add experiences that occur based on agent's actions,
--   and creates batches of experiences based on the mini-batch size for training.]] --
function Memory(maxMemory, discount)
    memory = {}
    -- Appends the experience to the memory.
    function memory.remember(memoryInput)
        table.insert(memory, memoryInput)
        if (#memory > maxMemory) then
            -- Remove the earliest memory to allocate new experience to memory.
            table.remove(memory, 1)
        end
    end
    function memory.getBatch(model, batchSize, nbActions, nbStates)
        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch).
        memoryLength = #memory
        chosenBatchSize = math.min(batchSize, memoryLength)
        inputs = torch.Tensor(chosenBatchSize, nbStates):zero()
        targets = torch.Tensor(chosenBatchSize, nbActions):zero()
        --Fill the inputs and targets up.
        for i = 1, chosenBatchSize do
            -- Choose a random memory experience to add to the batch.
            randomIndex = math.random(1, memoryLength)
            memoryInput = memory[randomIndex]
            target = model:forward(memoryInput.inputState)
            --Gives us Q_sa, the max q for the next state.
            nextStateMaxQ = torch.max(model:forward(memoryInput.nextState), 1)[1]
            if (memoryInput.gameOver) then
                target[memoryInput.action] = memoryInput.reward
            else
                -- reward + discount(gamma) * max_a' Q(s',a')
                -- We are setting the Q-value for the action to  r + γmax a’ Q(s’, a’). The rest stay the same
                -- to give an error of 0 for those outputs.
                target[memoryInput.action] = memoryInput.reward + discount * nextStateMaxQ
            end
            -- Update the inputs and targets.
            inputs[i] = memoryInput.inputState
            targets[i] = target
        end
        return inputs, targets
    end
    return memory
end

--[[ Runs one gradient update using SGD returning the loss.]] --
function trainNetwork(model, inputs, targets, criterion, sgdParams)
    loss = 0
    x, gradParameters = model:getParameters()
    function feval(x_new)
        gradParameters:zero()
        predictions = model:forward(inputs)
        loss = criterion:forward(predictions, targets)
        gradOutput = criterion:backward(predictions, targets)
        model:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    _, fs = optim.sgd(feval, x, sgdParams)
    loss = loss + fs[1]
    return loss
end

-- Create the base model.
model = nn.Sequential()
model:add(nn.Linear(nbStates, hiddenSize))
model:add(nn.ReLU())
model:add(nn.Linear(hiddenSize, hiddenSize))
model:add(nn.ReLU())
model:add(nn.Linear(hiddenSize, nbActions))

-- Params for Stochastic Gradient Descent (our optimizer).
sgdParams = {
    learningRate = 0.1,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}

-- Mean Squared Error for our loss function.
criterion = nn.MSECriterion()

env = CatchEnvironment(gridSize)
memory = Memory(maxMemory, discount)

winCount = 0
for i = 1, epoch do
    -- Initialise the environment.
    err = 0
    env.reset()
    isGameOver = false

    -- The initial state of the environment.
    currentState = env.observe()

    while (isGameOver ~= true) do
        action
        -- Decides if we should choose a random action, or an action from the policy network.
        if (randf(0, 1) <= epsilon) then
            action = math.random(1, nbActions)
        else
            -- Forward the current state through the network.
            q = model:forward(currentState)
            -- Find the max index (the chosen action).
            max, index = torch.max(q, 1)
            action = index[1]
        end
        -- Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
        if (epsilon > epsilonMinimumValue) then
            epsilon = epsilon * 0.999
        end
        nextState, reward, gameOver = env.act(action)
        if (reward == 1) then winCount = winCount + 1 end
        memory.remember({
            inputState = currentState,
            action = action,
            reward = reward,
            nextState = nextState,
            gameOver = gameOver
        })
        -- Update the current state and if the game is over.
        currentState = nextState
        isGameOver = gameOver

        -- We get a batch of training data to train the model.
        inputs, targets = memory.getBatch(model, batchSize, nbActions, nbStates)

        -- Train the network which returns the error.
        err = err + trainNetwork(model, inputs, targets, criterion, sgdParams)
    end
    print(string.format("Epoch %d : err = %f : Win count %d ", i, err, winCount))
end
torch.save(opt.savePath, model)
print("Model saved to " .. opt.savePath)