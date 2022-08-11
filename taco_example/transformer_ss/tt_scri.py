import torch

batch = [[1,2,3],[3,1],[1,1,1,1,1,1]]
input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)


print(input_lengths)



t1 = torch.tensor([[[1,2,3],[2,2,3]]])
print(t1.size())

for a in t1:
        print(a.size())




tt = torch.randn(1,100)
in_dim = 100
out_dim = 1
linear_layer = torch.nn.Linear(in_dim, out_dim, bias=True)
print(linear_layer.weight)
torch.nn.init.xavier_uniform_(
           linear_layer.weight,
           gain=torch.nn.init.calculate_gain("sigmoid"))
print(linear_layer.weight)

oo = linear_layer(tt)
print(oo)



ttt = torch.tensor([1,
        1])
sss = torch.le(ttt,0.5)
print(sss.to(torch.int32))