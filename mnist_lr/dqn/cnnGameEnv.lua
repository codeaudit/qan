require 'cunn'
require 'cutorch'
require 'dataset-mnist'
require 'optim'
local cnnGameEnv = torch.class('cnnGameEnv')

function cnnGameEnv:__init(opt)
	--add a rouletee
	--rouletee, lastloss read from file
	self.base = '../save/'
	function check(name)
		local f=io.open(name, "r")
		if f~=nil then io.close(f) return true else return false end
	end
	print('init gameenv')
	torch.manualSeed(os.time())
    self.model = self:create_mlp_model()
    self.model:cuda()
    self.parameters, self.gradParameters = self.model:getParameters()
    self.criterion = nn.ClassNLLCriterion():cuda()
    self.trsize = 20000
    self.tesize = 10000
    geometry = {32, 32}
    self.trainData = mnist.loadTrainSet(self.trsize, geometry)
    self.trainData:normalizeGlobal(mean, std)
    self.testData = mnist.loadTestSet(self.tesize, geometry)
    self.testData:normalizeGlobal(mean, std)
    self.classes = {'1','2','3','4','5','6','7','8','9','10'}
    self.confusion = optim.ConfusionMatrix(self.classes)
    self.batchsize = 10
    self.total_batch_number = math.ceil(self.trsize/self.batchsize)
    self.learningRate = 0.05 --1e-3
    self.weightDecay = 0
    self.momentum = 0
--	self.filter = torch.load(self.base..'cnnfilter1.t7')
	self.w1 = torch.load(self.base..'cnnfilter1.t7')
	self.w2 = torch.load(self.base..'cnnfilter2.t7')
	self.w3 = torch.load(self.base..'cnnfilter3.t7')
	self.w4 = torch.load(self.base..'cnnfilter4.t7')
    --self.filter2 = torch.load(self.base..'target_mlp2') --TODO
	--self.filter4 = torch.load(self.base..'target_mlp4')
    self.finalerr = 0.0001 
    self.epoch = 0
    self.batchindex = 1
    self.channel = 1
    self.er = 1
    self.mapping = {}
	self.terminal = false	
	self.havenewrecord = false
    self.datapointer = 1  --serve as a pointer, scan all the training data in one iteration.
    --self.model:float()
end

function cnnGameEnv:create_mlp_model()
    local model = nn.Sequential()
    -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
    model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
    -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
    model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    -- stage 3 : standard 2-layer MLP:
    model:add(nn.Reshape(64*2*2))
    model:add(nn.Linear(64*2*2, 200))
    model:add(nn.Tanh())
    model:add(nn.Linear(200, 10))
    model:add(nn.LogSoftMax())
    --[[model:add(nn.Reshape(1024))
    model:add(nn.Linear(1024, 256))
    model:add(nn.Tanh())
    model:add(nn.Linear(256,10))
    model:add(nn.LogSoftMax())]]
    return model
end

function cnnGameEnv:train()
    local dataset = self.trainData
    -- epoch tracker
    self.batchindex = self.batchindex or 1
    local time = sys.clock()
    local trainError = 0
    -- do one mini-batch
	local bsize = math.min(self.batchsize, dataset:size()-self.datapointer+1)
    local inputs = torch.CudaTensor(bsize, self.channel, 32, 32)
    local targets = torch.CudaTensor(bsize)
    for i = 1,bsize do
        local idx = self.datapointer
        local sample = dataset[idx]
        local input = sample[1]:clone()
        local _,target = sample[2]:clone():max(1)
        target = target:squeeze()
        inputs[i] = input
        targets[i] = target
        self.datapointer = self.datapointer + 1
    end
    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
        collectgarbage()
        if x ~= self.parameters then
            self.parameters:copy(x)
        end
        self.gradParameters:zero()
        local output = self.model:forward(inputs)
        local err = self.criterion:forward(output, targets)
        local df_do = self.criterion:backward(output, targets)
        self.model:backward(inputs, df_do)
        for i = 1,bsize do
            self.confusion:add(output[i], targets[i])
        end
        self.er = err
        return f,self.gradParameters
    end
    -- optimize on current mini-batch
    config = config or {learningRate = self.learningRate,
                  momentum = self.momentum,
                  learningRateDecay = 5e-7}
    optim.sgd(feval, self.parameters, config)
    print (self.confusion)
    local trainAccuracy = self.confusion.totalValid * 100 
    print(trainAccuracy)
    self.confusion:zero()
    return trainAccuracy, trainError
end

function cnnGameEnv:test()
    local dataset = self.testData
    local testError = 0
    local time = sys.clock()
    print('<trainer> on testing Set:')
    for t = 1,self.tesize do
       -- disp progress
       xlua.progress(t, self.tesize)
       -- get new sample
       local input = torch.CudaTensor(1,self.channel,32,32)
       input[1] = dataset.data[t]
       local target = dataset.labels[t]
       -- test sample
       local pred = self.model:forward(input[1])
       self.confusion:add(pred:view(10), target)
       -- compute error
       err = self.criterion:forward(pred, target)
       testError = testError + err
    end
    -- print confusion matrix
    print(self.confusion)
    local testAccuracy = self.confusion.totalValid * 100
    self.confusion:zero()
    return testAccuracy, testError
end

function cnnGameEnv:regression(targets, weights, layernum)
	if layernum == 1 then 
		input_neural_number = 32
		output_neural_number = 25
	elseif layernum == 2 then
		input_neural_number = 64
		output_neural_number = 800
	elseif layernum == 3 then
		input_neural_number = 200
		output_neural_number = 256
	elseif layernum == 4 then
		input_neural_number = 10
		output_neural_number = 200
	end

	--input_neural_number = 32
	--output_neural_number = 25
    --local targets = self.filter:cuda()
    --local weights = self.model:get(4).weight:cuda()
	targets = targets:cuda()
	weights = weights:cuda()
--    local regressweights = torch.CudaTensor(10, 256)
    local reg_data = torch.CudaTensor(input_neural_number, output_neural_number):fill(1)
    -- https://github.com/torch/nn/blob/master/doc/simple.md#nn.CMul
    reg_model = nn.Sequential()
    reg_model:add(nn.CMul(output_neural_number))
	reg_model:cuda()
    -- how to set weights into reg_model? set in line 156
    function regGradUpdate(reg_model, reg_x, reg_y, reg_criterion, regLearningRate)
        local reg_pred = reg_model:forward(reg_x)
        local reg_err = reg_criterion:forward(reg_pred, reg_y)
        local regGradCriterion = reg_criterion:backward(reg_pred, reg_y)
        reg_model:zeroGradParameters()
        reg_model:backward(reg_x, regGradCriterion)
        reg_model:updateParameters(regLearningRate)
        return reg_err
    end

    for i = 1, input_neural_number do
		reg_model:get(1).weight:copy(weights[i])
        -- do 3 iterations of regression
        for j = 1, 3 do
            err = regGradUpdate(reg_model, reg_data[i], targets[i], nn.MSECriterion():cuda(), 0.01)
        end
		weights[i]:copy(reg_model:get(1).weight)
    end
    -- need to set weights back here: set back in line 161
end



function cnnGameEnv:reward(verbose, filter, tstate)
	verbose = verbose or false
    local reward = 0
    if self.er then
        reward = 1 / math.abs(self.er - self.finalerr) 
    end
	if (verbose) then
        print ('finalerr: ' .. self.finalerr)
		if self.er then print ('err: ' .. self.er) end
		print ('reward: '.. reward)
	end
	print ('err: ' .. self.er)
	print ('reward is: ' .. reward)
	return reward
end

function cnnGameEnv:getActions()
	local gameActions = {}
	for i=1,3 do
		gameActions[i] = i
	end
	return gameActions
end

function cnnGameEnv:nObsFeature()

end

function cnnGameEnv:getState(verbose) --state is set in cnn.lua
	verbose = verbose or false
	--return state, reward, term
	local tstate = self.model:get(1).weight 
	local size = tstate:size()[1] * tstate:size()[2]  
	print(size)
    local filter = self.filter
	local reward = self:reward(verbose, filter, tstate)
	if self.terminal == true then
		self.terminal = false
		return tstate, reward, true
	else
		return tstate, reward, false
	end	
end

function cnnGameEnv:step(action, tof)
	print('step')
	io.flush()

	--[[
		action 1: increase
		action 2: decrease
		action 3: unchanged	
	]]
	local delta = 0.005
	local minlr = 0.005
	local maxlr = 1.0
	local outputtrain = 'train_lr.log'--'basetrain.log'--'baseline_raw_train.log'
	local outputtest = 'test_lr.log'--'basetest.log'--'baseline_raw_test.log'

    if (action == 1) then 
        self.learningRate = math.min(self.learningRate + delta, maxlr);
    elseif (action == 2) then 
        self.learningRate = math.max(self.learningRate - delta, minlr);
    end

    print('<trainer> on training set:' .. 'epoch #' .. self.epoch .. ', batchindex ' .. self.batchindex)
    trainAcc, trainErr = self:train()
    self.trainAcc = self.trainAcc or 0
    self.trainAcc = self.trainAcc + trainAcc
	--local w2 = self.model:get(2).weight
    --local w4 = self.model:get(4).weight
	local ww = self.model:get(1).weight
	print('batchindex = '.. self.batchindex)
	print('totalbatchindex = '.. self.total_batch_number)

    if self.epoch % 20 <= 5 then   --Let mlp train freely after 10 epoches.
		local w1 = self.model:get(1).weight
		local w2 = self.model:get(4).weight
		local w3 = self.model:get(8).weight
		local w4 = self.model:get(10).weight
		self:regression(self.w1, w1, 1)
		self:regression(self.w2, w2, 2)
		self:regression(self.w3, w3, 3)
		self:regression(self.w4, w4, 4)
    end
    if self.batchindex == self.total_batch_number then
        self.datapointer = 1 --reset the pointer
        self.trainAcc = self.trainAcc / self.total_batch_number
        print ('trainAcc = ' .. self.trainAcc)
        os.execute('echo ' .. self.trainAcc .. ' >> logs/' .. outputtrain)
        self.trainAcc = 0
        testAcc,  testErr = self:test()
        os.execute('echo ' .. testAcc .. ' >> logs/' .. outputtest)
        self.batchindex = 1
        self.epoch = self.epoch + 1
        print('epoch = ' .. self.epoch)
        if self.epoch > 0 and self.epoch % 20 == 0 then
			--[[local w1 = self.model:get(1).weight
			local w2 = self.model:get(4).weight
			local w3 = self.model:get(8).weight
			local w4 = self.model:get(10).weight
			torch.save('../save/cnnfilter1.t7', w1)
			torch.save('../save/cnnfilter2.t7', w2)
			torch.save('../save/cnnfilter3.t7', w3)
			torch.save('../save/cnnfilter4.t7', w4)]]
			self.learningRate = 0.05
			self.terminal = true
            print ('epoch 100!>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<')
            --reset model
            self.model = self:create_mlp_model()
            self.model:cuda()
            self.parameters, self.gradParameters = self.model:getParameters()
        end
    end
    self.batchindex = self.batchindex + 1
	return self:getState()
end

function cnnGameEnv:nextRandomGame()

end

function cnnGameEnv:newGame()

end

