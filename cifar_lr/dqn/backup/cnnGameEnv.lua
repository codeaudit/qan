
local cnnGameEnv = torch.class('cnnGameEnv')

function cnnGameEnv:__init(opt)
	--add a rouletee
	--rouletee, lastloss read from file
	self.rnum = 7
	self.base = '/mnt/ramdisk/'
	local rouletee_path = self.base .. 'save/ROULETEE'
	--local lastloss_path = self.base .. 'save/LASTLOSS'
	function check(name)
		local f=io.open(name, "r")
		if f~=nil then io.close(f) return true else return false end
	end
	if check(rouletee_path) then
		self.rouletee = torch.load(rouletee_path)
	else 
		self.rouletee = nil
	end
	if (self.rouletee == nil or self.rouletee:storage():size() ~= self.rnum) then
		local r = {}
		for i=1,self.rnum do
			r[i] = 1
		end
		self.rouletee = torch.Tensor(r)
	end
	--if check(lastloss_path) then
	--	self.lastloss = torch.load(lastloss_path) 
	--else 
	--	self.lastloss = nil
	--end
	--self.rouletee = {nil,nil,nil,nil}
	--self.lastloss = nil
	self.batchsize = 128
	print('init gameenv')
	self.c = {}
	for i=1,10 do
		self.c[i] = torch.load(self.base..'save/CLASSIFY'..i)
	end
	torch.manualSeed(os.time())
end

function cnnGameEnv:save_temp()
	--torch.save('/home/jie/lzc/squeeze_data/atari/save/ROULETEE', self.rouletee)
	--torch.save('/home/jie/lzc/squeeze_data/atari/save/LASTLOSS', self.lastloss)
end

function cnnGameEnv:reward(verbose, filter, tstate)
	verbose = verbose or false
    local lasterr_path = self.base..'save/LASTERR'
    local err_path = self.base..'save/ERR'
    local lasterr = nil
    local er = nil
	if check(lasterr_path) then
        lasterr = torch.load(self.base..'save/LASTERR')
	end
	if check(err_path) then
        er = torch.load(self.base..'save/ERR')
	end
    local reward = 0
    --if (lasterr and lasterr-er > 0) then
    --    reward = lasterr - er
    --end
    if lasterr and er then
        reward = (lasterr - er) * 100000
    end
    torch.save(self.base .. 'save/LASTERR', er)
    ----------------
	if (verbose) then
		print ('err: ' .. er)
		if lasterr then print ('lasterr: ' .. lasterr) end
		print ('reward: '.. reward)
	end
	--self.lastloss = loss
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

function cnnGameEnv:getState(verbose) --state is set in cnn.lua
	verbose = verbose or false
	--return state, reward, term
	local tstate = torch.load(self.base..'save/NEXT_STATE')
	local size = tstate:size()[1] * tstate:size()[2]  
	print(size)
	--tstate = tstate:view(size)
	--[[local state = {}
	for i=1,size do
	state[i] = tstate[i]
	end]]
    local filter = torch.load(self.base..'save/FILTER')
	local reward = self:reward(verbose, filter, tstate)
	return tstate, reward, false
end

function cnnGameEnv:step(action, tof)
	print('step')
	io.flush()
	--subtitue rouletee[4], add action
	for i=self.rnum,2,-1 do
		self.rouletee[i] = self.rouletee[i-1]
	end
	--self.rouletee[4] = self.rouletee[3]
	--self.rouletee[3] = self.rouletee[2]
	--self.rouletee[2] = self.rouletee[1]
	self.rouletee[1] = action
	--select mini-batch according to actions
	--actions number are classes number
	local num = 0
	for i=1,self.rnum do
		if (self.rouletee[i]) then
			num = num + 1
		end
	end
	local esize = math.floor(self.batchsize / num)
	--get datapoints from /home/jie/class2index/c?.dat according to each_size
	local res = {}
	--sample self.batchsize-esize*(num-1) from class r[1]
	local c1 = self.rouletee[1]
	local r1size = self.batchsize-esize*(num-1)
	--print("r1size = "..r1size)
	--print("esize = "..esize)
	print('actions are: ')
	for i=1,self.rnum do
		print(self.rouletee[i])
	end
	io.flush()
--print (self.c[c1]:storage():size())
	local shuffle0 = torch.randperm(self.c[c1]:storage():size())
	--print (shuffle0)
	for j=1,r1size do
		--print (self.c[c1][shuffle0[j]])
		res[#res+1] = self.c[c1][shuffle0[j]]
	end
	--sample esize from class r[i]
	for i=2,self.rnum do
		if (self.rouletee[i]) then
			local c = self.rouletee[i]
			local shuffle = torch.randperm(self.c[c]:storage():size())
			for j=1,esize do
				res[#res+1] = self.c[c][shuffle[j]]
			end
		end
	end
	local batch = torch.LongTensor(res) --batch_indices to learn
	--save the batch indices in ../save/ACTION for CNN
	print('finish step, write ACTION into save/ACTION')
	io.flush()
	torch.save(self.base..'save/ACTION', batch)
	torch.save(self.base..'save/ROULETEE', self.rouletee)
	torch.save(self.base..'save/LASTLOSS', self.lastloss)
	--send batch to CNN to train, and receive loss, then compute reward!
	--write CNN train interface
	return self:getState()
end

function cnnGameEnv:nextRandomGame()

end

function cnnGameEnv:newGame()

end

