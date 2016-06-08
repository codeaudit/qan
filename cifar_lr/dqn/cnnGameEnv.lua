require 'cunn'
require 'cutorch'
require 'dataset-mnist'
require 'optim'
dofile 'provider.lua'
require 'BatchFlip'
local c = require 'trepl.colorize'
local cnnGameEnv = torch.class('cnnGameEnv')

local function cast(t)
    return t:cuda()
end
function cnnGameEnv:__init(opt)
	--add a rouletee
	--rouletee, lastloss read from file
	self.rnum = 10
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
    self.criterion = cast(nn.CrossEntropyCriterion())--nn.ClassNLLCriterion():cuda()
    self.trsize = 50000
    self.tesize = 10000
    geometry = {32, 32}
	--self.trainData = torch.load(self.base..'traindatagray.t7')
	--self.testData = torch.load(self.base..'testdatagray.t7')
	provider = torch.load('../save/provider.t7')
	--provider:normalize()
	provider.trainData.data = provider.trainData.data:float()
	provider.testData.data = provider.testData.data:float()
	self.trainData = provider.trainData
	self.testData = provider.testData
    self.classes = {'1','2','3','4','5','6','7','8','9','10'}
    self.confusion = optim.ConfusionMatrix(self.classes)
    self.batchsize = 128
	self.game_epoch = 30
    self.total_batch_number = math.ceil(self.trsize/self.batchsize)
	local v = {}
	for k=1,10 do v[k]=k end
    self.filter = torch.load(self.base..'target_mlp1'):view(64,27):index(1, torch.LongTensor(v))
	print(self.filter:size())
    --self.filter2 = torch.load(self.base..'target_mlp2') 
    --self.filter4 = torch.load(self.base..'target_mlp4') 
    self.finalerr = 0.05 --torch.load(self.base..'FINALERR')
    self.epoch = 0
    self.batchindex = 1
    self.channel = 1
    self.er = 1
	self.finalerr = 0.0001
    self.mapping = {}
	self.terminal = false	
	self.havenewrecord = false
    self.datapointer = 1  --serve as a pointer, scan all the training data in one iteration.
	self.epoch_step = 25
	self.optimState = {
		learningRate = 1,
		weightDecay = 0.0005,
		momentum = 0.9,
		learningRateDecay = 1e-7
	}
    self.indices = torch.randperm(self.trainData.data:size(1)):long():split(self.batchsize)
    -- remove last element so that all the batches have equal size
    self.indices[#self.indices] = nil
	self.batchpointer = 1
	self.delta = 0.1
end

function cnnGameEnv:create_mlp_model()
	local model = nn.Sequential()
	--model:add(nn.BatchFlip():float())
	model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
	model:add(cast(dofile('models/vgg_bn_drop.lua')))
	model:get(2).updateGradInput = function(input) return end
	
	print(model)
	return model
end

function cnnGameEnv:train()
  self.model:training()

  -- drop learning rate every "epoch_step" epochs
  if self.epoch % self.epoch_step == 0 then 
	  self.optimState.learningRate = self.optimState.learningRate/2 
	  self.delta = self.delta/2
  end
  
  local targets = cast(torch.FloatTensor(self.batchsize))

  local tic = torch.tic()
  --for t,v in ipairs(indices) do
    xlua.progress(self.batchpointer, #self.indices)

	local v = self.indices[self.batchpointer]
	self.batchpointer = self.batchpointer + 1
    local inputs = self.trainData.data:index(1,v)
    targets:copy(self.trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= self.parameters then self.parameters:copy(x) end
      self.gradParameters:zero()
      
      local outputs = self.model:forward(inputs)
      local f = self.criterion:forward(outputs, targets)
	  self.er = f
      local df_do = self.criterion:backward(outputs, targets)
      self.model:backward(inputs, df_do)

      self.confusion:batchAdd(outputs, targets)

      return f,self.gradParameters
    end
    optim.sgd(feval, self.parameters, self.optimState)
  --end

  self.confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        self.confusion.totalValid * 100, torch.toc(tic)))

  train_acc = self.confusion.totalValid * 100

  self.confusion:zero()
  return train_acc
end

function cnnGameEnv:test()
  -- disable flips, dropouts and batch normalization
  self.model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  for i=1,self.testData.data:size(1),bs do
    local outputs = self.model:forward(self.testData.data:narrow(1,i,bs))
    self.confusion:batchAdd(outputs, self.testData.labels:narrow(1,i,bs))
  end

  self.confusion:updateValids()
  local testacc = self.confusion.totalValid * 100
  print('Test accuracy:', testacc)
  
  self.confusion:zero()
  return testacc
end

function cnnGameEnv:regression(targets, weights)
	input_neural_number = 10
	output_neural_number = 27
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
	--print(self.model:get(3):get(54):get(6))
	local v = {}
	for k=1,10 do v[k]=k end
	local w = self.model:get(2):get(1).weight:view(64,27):index(1, torch.LongTensor(v))
	local tstate = w
	--local tstate = self.model:get(2):get(54):get(6).weight 
	print(tstate:size())
	local size = tstate:size()[1] * tstate:size()[2]  
    local filter = self.filter4
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
	local minlr = 0.001
	local maxlr = 1.0
	local outputtrain = 'vgg_raw_train.log'
	local outputtest = 'vgg_raw_test.log'

    if (action == 1) then 
        self.optimState.learningRate = math.min(self.optimState.learningRate + self.delta, maxlr);
    elseif (action == 2) then 
        self.optimState.learningRate = math.max(self.optimState.learningRate - self.delta, minlr);
    end
	print('learningRate = '..self.optimState.learningRate)

    print('<trainer> on training set:' .. 'epoch #' .. self.epoch .. ', batchindex ' .. self.batchindex)
    trainAcc = self:train()
    self.trainAcc = self.trainAcc or 0
    self.trainAcc = self.trainAcc + trainAcc
	--local w2 = self.model:get(2).weight
    --local w4 = self.model:get(4).weight
	--print('batchindex = '.. self.batchindex)
	--print('totalbatchindex = '.. self.total_batch_number)
	print('batchpointer = '.. self.batchpointer)
	print('totalbatchnumber = '.. #self.indices)

	local v = {}
	for k=1,10 do v[k]=k end
	local w = self.model:get(2):get(1).weight:view(67,27):index(1,torch.LongTensor(v))

    if self.epoch % self.game_epoch <= 30 then   --Let mlp train freely after 10 epoches.
    	self:regression(self.filter, w) 
    end
    if (self.batchpointer == #self.indices) then
        self.datapointer = 1 --reset the pointer
        self.batchpointer = 1 --reset the batch pointer
        self.trainAcc = self.trainAcc / #self.indices
        print ('trainAcc = ' .. self.trainAcc)
        os.execute('echo ' .. self.trainAcc .. ' >> logs/' .. outputtrain)
        self.trainAcc = 0
        testAcc = self:test()
		self.maxtestAcc = self.maxtestAcc or 96.52
		if testAcc > self.maxtestAcc then
			self.maxtestAcc = testAcc
			self.havenewrecord = true
			os.execute('echo ' .. testAcc .. ' >> logs/record.log')
			self.optimalmlp = torch.CudaTensor(10,27):copy(w)
			--self.optimalmlp2 = torch.CudaTensor(256,1024):copy(w2)
			--self.optimalmlp4 = torch.CudaTensor(10,256):copy(w4)
		end
        os.execute('echo ' .. testAcc .. ' >> logs/' .. outputtest)
        self.batchindex = 1
        self.batchpointer = 1
        self.epoch = self.epoch + 1
        print('epoch = ' .. self.epoch)
        if self.epoch > 0 and self.epoch % self.game_epoch == 0 then
			self.optimState.learningRate = 1
			self.terminal = true
            print ('epoch 100!>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<')
			--NOTE: update with the optimal mlp if have new acc record in the training process!!!
			if self.havenewrecord == true then
				self.havenewrecord = false
				self.filter = self.optimalmlp
				--self.filter2 = self.optimalmlp2 --self.model:get(2).weight
				--self.filter4 = self.optimalmlp4 --self.model:get(4).weight
			end
			--otherwise maintain old mlp
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

