require 'nn'
require 'optim'
require 'cunn'
require 'cutorch'
require 'dataset-mnist'

local cnnGameEnv = torch.class('cnnGameEnv')

function cnnGameEnv:__init(opt)
	--add a rouletee
	--rouletee, lastloss read from file
	self.rnum = 10
	self.base = '../save/'
	function check(name)
		local f=io.open(name, "r")
		if f~=nil then io.close(f) return true else return false end
	end
    local r = {}
    for i=1,self.rnum do
        r[i] = 1
    end
    self.rouletee = torch.Tensor(r)
	print('init gameenv')
	self.c = {}
	for i=1,10 do
		self.c[i] = torch.load(self.base..'MNIST'..i) --TODO
	end
	torch.manualSeed(os.time())
    self.model = self:create_mlp_model()
    self.model:cuda()
    self.parameters, self.gradParameters = self.model:getParameters()
    self.criterion = nn.ClassNLLCriterion():cuda()
    --self.trainData = torch.load('../mnist/train_32x32_20000.t7') --TODO
    --self.testData = torch.load('../mnist/test_32x32.t7', 'ascii')
    self.trsize = 20000
    self.tesize = 10000
    geometry = {32, 32}
    self.trainData = mnist.loadTrainSet(self.trsize, geometry)
    self.trainData:normalizeGlobal(mean, std)
    self.testData = mnist.loadTestSet(self.tesize, geometry)
    self.testData:normalizeGlobal(mean, std)
    self.classes = {'1','2','3','4','5','6','7','8','9','10'}
    self.confusion = optim.ConfusionMatrix(self.classes)
    self.batchsize = 128
    self.total_batch_number = math.floor(self.trsize/self.batchsize)
    self.learningRate = 0.05 --1e-3
    self.weightDecay = 0
    self.momentum = 0
    self.filter = torch.load(self.base..'cnnfilter1.t7') 
    self.finalerr = 0 --torch.load(self.base..'FINALERR')
    self.epoch = 0
    self.batchindex = 1
    self.channel = 1
    self.er = 1
    self.mapping = {}
    for i=1,10 do self.mapping[i] = 0 end
	self.randindex = {}
	self.pointer = {}
	for i=1,10 do
		self.randindex[i] = torch.randperm(self.c[i]:storage():size())
		self.pointer[i] = 1
	end
	self.terminal = false	
	self.havenewrecord = false
	self.max_epoch = 40
	self.dp = 1
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
    --[[model:add(nn.Reshape(3*32*32))
    model:add(nn.Linear(3*32*32, 1*32*32))
    model:add(nn.Dropout())
    model:add(nn.Tanh())
    model:add(nn.Linear(1*32*32, 512))
    model:add(nn.Dropout())
    model:add(nn.Tanh())
    model:add(nn.Linear(512, 10))
    model:add(nn.LogSoftMax())]]
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
	
      local indices = self.batch 
	  local bsize = self.batch:storage():size()
      local inputs = torch.CudaTensor(bsize, self.channel, 32, 32)
      local targets = torch.CudaTensor(bsize)
      for i = 1,bsize do
         local idx = indices[i] 
         --[[local input = dataset.data[idx]:clone()
         local target = dataset.labels[idx]
         inputs[i] = input
         targets[i] = target]]
         local sample = dataset[idx]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[i] = input
         targets[i] = target
      end
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
	 collectgarbage()
         -- get new parameters
         if x ~= self.parameters then
            self.parameters:copy(x)
         end
         self.gradParameters:zero()
         -- evaluate function for complete mini batch
         --for i = 1,self.batchsize do
            -- estimate f
            local output = self.model:forward(inputs)
            local err = self.criterion:forward(output, targets)
            --f = f + err
            -- estimate df/dW
            local df_do = self.criterion:backward(output, targets)
            self.model:backward(inputs, df_do)
            -- update confusion
	 for i = 1,bsize do
         --self.confusion:add(output:view(10), targets[i])
         self.confusion:add(output[i], targets[i])
	 end
         --end
         -- normalize gradients and f(X)
         --self.gradParameters:div(self.batchsize)
         --f = f/self.batchsize
	 self.er = err
         --trainError = trainError + f
         self.loss = err --torch.save(base..'LOSS', f)
         -- return f and df/dX
         return f,self.gradParameters
      end
      -- optimize on current mini-batch
      config = config or {learningRate = self.learningRate,
                          weightDecay = self.weightDecay,
                          momentum = self.momentum,
                          learningRateDecay = 5e-7}
      optim.sgd(feval, self.parameters, config)
   --trainError = trainError / math.floor(dataset:size()/self.batchsize)
   -- print confusion matrix
   print (self.confusion)
   local trainAccuracy = self.confusion.totalValid * 100 
   print(trainAccuracy)
   --os.execute('echo ' .. trainAccuracy .. ' >> logs/train.log')
   self.confusion:zero()
   -- next epoch
   --epoch = epoch + 1
   return trainAccuracy, trainError
end

function cnnGameEnv:test()
    local dataset = self.testData
    -- local vars
    local testError = 0
    local time = sys.clock()
    -- test over given dataset
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

function cnnGameEnv:regression(targets, weights)
print(targets:size())
print(weights:size())
	input_neural_number = 32
	output_neural_number = 25
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

function cnnGameEnv:getState(verbose) --state is set in cnn.lua
	verbose = verbose or false
	--return state, reward, term
	local tstate = self.model:get(1).weight --torch.load(self.base..'save/NEXT_STATE')
	local size = tstate:size()  
	print(size)
    local filter = self.filter --torch.load(self.base..'save/FILTER')
	local reward = self:reward(verbose, filter, tstate)
	if self.terminal == true then
		self.terminal = false
		return tstate, reward, true
	else
		return tstate, reward, false
	end	
end

function cnnGameEnv:reward(verbose, filter, tstate)
	verbose = verbose or false
    local reward = 0
    if self.er then
        --reward = 1 / (self.er - self.finalerr) 
        reward = 1 / (self.er - 0.001) 
    end
    ----------------
	if (verbose) then
        print ('finalerr: ' .. self.finalerr)
		if self.er then print ('err: ' .. self.er) end
		print ('reward: '.. reward)
	end
	--self.lastloss = loss
	print ('err: ' .. self.er)
	print ('reward is: ' .. reward)
	return reward
end

function cnnGameEnv:getActions()
	local gameActions = {}
	for i=1,10 do
		gameActions[i] = i
	end
	return gameActions
end

function cnnGameEnv:nObsFeature()
end

function cnnGameEnv:step(action, tof)
	print('step')
	io.flush()
	for i=self.rnum,2,-1 do
		self.rouletee[i] = self.rouletee[i-1]
	end
	self.rouletee[1] = action
	--select mini-batch according to actions
	--actions number are classes number
	local num = 0
	for i=1,self.rnum do
		if (self.rouletee[i]) then
			num = num + 1
			self.mapping[self.rouletee[i]] = self.mapping[self.rouletee[i]] + 1
		end
	end
	local esize = math.floor(self.batchsize / num)
	local res = {}
	--sample self.batchsize-esize*(num-1) from class r[1]
	local c1 = self.rouletee[1]
	local r1size = self.batchsize-esize*(num-1)
	print('esize='..esize)
	print('r1size='..r1size)
	--print("r1size = "..r1size)
	--print("esize = "..esize)
	print('actions are: ')
	for i=1,self.rnum do
		print(self.rouletee[i])
	end
	io.flush()
	--local shuffle0 = torch.randperm(self.c[c1]:storage():size())
	--print (shuffle0)
	for j=1,r1size do
		--print (self.c[c1][shuffle0[j]])
		if self.pointer[c1] <= self.c[c1]:storage():size() then
			--res[#res+1] = self.c[c1][shuffle0[j]]
			res[#res+1] = self.c[c1][self.randindex[c1][self.pointer[c1]]]
			self.pointer[c1] = self.pointer[c1] + 1  --TODO should reset to 0 when restart game
		end
	end
	--sample esize from class r[i]
	for i=2,self.rnum do
		if (self.rouletee[i]) then
			local c = self.rouletee[i]
			--local shuffle = torch.randperm(self.c[c]:storage():size())
			for j=1,esize do
				--res[#res+1] = self.c[c][shuffle[j]]
				if self.pointer[c] <= self.c[c]:storage():size() then
					res[#res+1] = self.c[c][self.randindex[c][self.pointer[c]]]
					self.pointer[c] = self.pointer[c] + 1  --TODO should reset to 0 when restart game
				end
			end
		end
	end
	--print(res)
	--print('end'..nil)
	if #res < self.batchsize then
		for i=1,10 do
			while self.pointer[i] <= self.c[i]:storage():size() and #res < self.batchsize do
				res[#res+1] = self.c[i][self.randindex[i][self.pointer[i]]]
				self.pointer[i] = self.pointer[i] + 1  --TODO should reset to 0 when restart game
			end
			if #res == self.batchsize then break end
		end
	end
	self.batch = torch.LongTensor(res) --batch_indices to learn
	io.flush()

    print('<trainer> on training set:' .. 'epoch #' .. self.epoch .. ', batchindex ' .. self.batchindex)
    trainAcc, trainErr = self:train()
    self.trainAcc = self.trainAcc or 0
    self.trainAcc = self.trainAcc + trainAcc
    local w = self.model:get(1).weight
	if self.epoch % self.max_epoch <= 5 then   --Let mlp train freely after 10 epoches.
		--self:regression(self.filter2, self.model:get(2).weight, 2)
    	--self:regression(self.filter4, self.model:get(4).weight, 4)
    	self:regression(self.filter, self.model:get(1).weight)
	end
    if (self.batchindex == self.total_batch_number) then
        --for i=1,10 do os.execute('echo '..self.mapping[i]..' >> logs/mapping.txt') end
        self.trainAcc = self.trainAcc / self.total_batch_number
        print ('trainAcc = ' .. self.trainAcc)
        --os.execute('echo ' .. self.trainAcc .. ' >> logs/train_minibatch_bs128_last_testing_40epoch_baseline.log')
        os.execute('echo ' .. self.trainAcc .. ' >> logs/train_minibatch_bs128.log')
        self.trainAcc = 0
        testAcc,  testErr = self:test()
        --os.execute('echo ' .. testAcc .. ' >> logs/test_minibatch_bs128_last_testing_40epoch_baseline.log')
        os.execute('echo ' .. testAcc .. ' >> logs/test_minibatch_bs128.log')
        self.batchindex = 1
        self.epoch = self.epoch + 1
        print('epoch = ' .. self.epoch)
		for i=1,10 do
			self.pointer[i] = 1
			--TODO randperm again?
		end
        if self.epoch > 0 and self.epoch % self.max_epoch == 0 then
			self.learningRate = 0.05
			self.terminal = true
            print ('epoch 100!>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<')
            --update target MLP
            --reset model
            self.model = self:create_mlp_model()
            self.model:cuda()
            self.parameters, self.gradParameters = self.model:getParameters()
        end
    end
    self.batchindex = self.batchindex + 1
	--send batch to CNN to train, and receive loss, then compute reward!
	--write CNN train interface
	return self:getState()
end

function cnnGameEnv:nextRandomGame()

end

function cnnGameEnv:newGame()

end

