----------------------------------------------------------------------
-- This script shows how to train different models on the CIFAR
-- dataset, using multiple optimization techniques (SGD, ASGD, CG)
--
-- This script demonstrates a classical example of training
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem.
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'nn'
require 'optim'
require 'image'
--require 'cunn'
--require 'cutorch'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('CIFAR Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-model', 'mlp', 'type of model to train: convnet | mlp | linear')
cmd:option('-full', true, 'use full dataset (50,000 samples)')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 1, 'nb of threads to use')
cmd:option('-batchindex', 0, 'batch index')
cmd:option('-epoch', 0, 'epoch index')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(opt.seed)

base = '/mnt/ramdisk/save/'
-- threads
torch.setnumthreads(opt.threads)
--print('<torch> set nb of threads to ' .. opt.threads)

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()
   --model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'))

   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network
      ------------------------------------------------------------
      -- stage 1 : mean+std normalization -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMap(nn.tables.random(3,16,1), 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 2 : filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMap(nn.tables.random(16, 256, 4), 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer neural network
      model:add(nn.Reshape(256*5*5))
      model:add(nn.Linear(256*5*5, 128))
      model:add(nn.Tanh())
      model:add(nn.Linear(128,#classes))
      model:add(nn.LogSoftMax())
      ------------------------------------------------------------

   elseif opt.model == 'mlp' then
      ------------------------------------------------------------
      -- regular 2-layer MLP
      ------------------------------------------------------------
      model:add(nn.Reshape(3*32*32))
      model:add(nn.Linear(3*32*32, 1*32*32))
      model:add(nn.Dropout())
      model:add(nn.Tanh())
      model:add(nn.Linear(1*32*32, 512))
      model:add(nn.Dropout())
      model:add(nn.Tanh())
      model:add(nn.Linear(512, #classes))
      model:add(nn.LogSoftMax())
      ------------------------------------------------------------

   elseif opt.model == 'linear' then
      ------------------------------------------------------------
      -- simple linear model: logistic regression
      ------------------------------------------------------------
      model:add(nn.Reshape(3*32*32))
      model:add(nn.Linear(3*32*32,#classes))
      ------------------------------------------------------------

   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
   --model = nn.Sequential()
   --model:read(torch.DiskFile(opt.network))
end

--model = model:cuda()
-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
--print('<cifar> using model:')
--print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
--criterion = nn.ClassNLLCriterion():cuda()
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   trsize = 50000
   tesize = 10000
else
   trsize = 2000
   tesize = 1000
end



trainData = torch.load('traindata.t7')--:cuda()
testData  = torch.load('testdata.t7')--:cuda()


----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
accLogger = optim.Logger(paths.concat(opt.save, 'accuracy.log'))
errLogger = optim.Logger(paths.concat(opt.save, 'error.log'   ))

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   local trainError = 0

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   --for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      --xlua.progress(t, dataset:size())

      -- create mini batch
      --local inputs = torch.CudaTensor(opt.batchSize, 3, 32, 32)
      --local targets = torch.CudaTensor(opt.batchSize)
      local inputs = torch.Tensor(opt.batchSize, 3, 32, 32)
      local targets = torch.Tensor(opt.batchSize)
      --[[for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local input = dataset.data[i]
         local target = dataset.labels[i]
         table.insert(inputs, input)
         table.insert(targets, target)
      end]]
      local indices = torch.load(base..'ACTION')
      for i = 1, 128 do
         local idx = indices[i] 
         local input = dataset.data[idx]:clone()
         local target = dataset.labels[idx]
         inputs[i] = input
         targets[i] = target
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- f is the average of all criterions
         local f = 0

         -- evaluate function for complete mini batch
         for i = 1,opt.batchSize do
            -- estimate f
            local output = model:forward(inputs[i])
            local err = criterion:forward(output, targets[i])
            f = f + err

            -- estimate df/dW
            local df_do = criterion:backward(output, targets[i])
            model:backward(inputs[i], df_do)

            -- update confusion
            confusion:add(output, targets[i])

            -- visualize?
            if opt.visualize then
               display(inputs[i])
            end
         end

         -- normalize gradients and f(X)
         gradParameters:div(opt.batchSize)
         f = f/opt.batchSize
         trainError = trainError + f
         torch.save(base..'LOSS', f)

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'CG' then
         config = config or {maxIter = opt.maxIter}
         optim.cg(feval, parameters, config)

      elseif opt.optimization == 'LBFGS' then
         config = config or {learningRate = opt.learningRate,
                             maxIter = opt.maxIter,
                             nCorrection = 10}
         optim.lbfgs(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
         config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = 5e-7}
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = opt.learningRate,
                             t0 = nbTrainingPatches * opt.t0}
         _,_,average = optim.asgd(feval, parameters, config)

      else
         error('unknown optimization method')
      end
   --end

   -- train error
   trainError = trainError / math.floor(dataset:size()/opt.batchSize)

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print (confusion)
   local trainAccuracy = confusion.totalValid * 100
   os.execute('echo ' .. trainAccuracy .. ' >> logs/train.log')
   confusion:zero()

   -- save/log current net
   --local filename = paths.concat(opt.save, 'cifar.net')
   --os.execute('mkdir -p ' .. paths.dirname(filename))
   --if paths.filep(filename) then
      --os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   --end
   --print('<trainer> saving network to '..filename)
   --torch.save(filename, model)

   -- next epoch
   --epoch = epoch + 1

   return trainAccuracy, trainError
end

-- test function
function test(dataset)
   -- local vars
   local testError = 0
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size() do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- get new sample
      local input = torch.Tensor(1,3,32,32)
      input[1] = dataset.data[t]
      local target = dataset.labels[t]

      -- test sample
      local pred = model:forward(input[1])
      confusion:add(pred, target)

      -- compute error
      err = criterion:forward(pred, target)
      testError = testError + err
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- testing error estimation
   testError = testError / dataset:size()

   -- print confusion matrix
   print(confusion)
   local testAccuracy = confusion.totalValid * 100
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end

   return testAccuracy, testError
end

function regression()
    local target = torch.load(base..'MLP')
    local weight = model:get(8).weight
    local regressweight = torch.Tensor(10, 512)
    local regresspos = torch.Tensor(10)
    local visited = torch.Tensor(10):fill(0)
    local er = 0
    for i = 1, 10 do
        local t = target[i]
        local minmse = 1000
        local minidx = 0
        for j = 1, 10 do    
            if visited[j]==0 then
                local w = weight[j]
                local mse = 0
                for k = 1, 512 do
                    mse = mse + math.pow(w[k]-t[k], 2)
                end
                if mse < minmse then
                    minmse = mse 
                    minidx = j
                end
            end
        end
        er = er + minmse
        visited[minidx] = 1
        regresspos[i] = minidx
    end
    er = math.sqrt(er)
    torch.save(base..'ERR', er)
    for i = 1, 10 do
        regressweight[i]:copy(weight[regresspos[i]])
    end
    weight = regressweight
end
----------------------------------------------------------------------
-- and train!
--
--while true do
   -- train/test
   trainAcc, trainErr = train(trainData)
   regression()
   local w = model:get(8).weight
   print(trainAcc)
   if (opt.batchindex == 389) then
      testAcc,  testErr  = test (testData)
      os.execute('echo ' .. testAcc .. ' >> logs/test.log')
      if opt.epoch > 0 and opt.epoch % 100 == 0 then
          torch.save(base..'MLP', w:float())
      end
   end
   --testAcc,  testErr  = test (testData)
--end

torch.save(base..'NEXT_STATE', w:float())
torch.save(base..'CIFAR', model)




