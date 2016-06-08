a = torch.load('traindata.t7')
print (a)
l = a.labels
size = l:storage():size()
print (size)
c = {}
for i=1,10 do
   c[i] = {}
end
for i = 1,size do
   local idx = l[i]
   c[idx][#c[idx]+1] = i
end

for i=1,10 do
   torch.save('CLASSIFY'..i, torch.LongTensor(c[i]))
end
