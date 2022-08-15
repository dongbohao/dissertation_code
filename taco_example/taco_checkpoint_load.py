import torch
import torch.nn as nn
import torch.optim as optim
from glottal_mel_model_v5 import TfModel,ntokens,emsize,nhead,d_hid,nlayers,dropout,ag,optimizer,testset_loader,criterion,print_spectrogram
import data_function
from tacotron2_common.utils import to_gpu
from ABCD import print_fre


model = TfModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, ag.n_mel_channels, batch_first=True, ag = ag)

#checkpoint = torch.load(r"C:\Users\xelloss\Downloads\checkpoint_v5_10ep_lr1-3.pt")
checkpoint = torch.load(r"C:\Users\xelloss\Downloads\checkpoint_v5_10ep.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print("Training loss", loss)
#model.eval()

def tt_dataset():
    for i, batch in enumerate(testset_loader):
        x, y, num_items = data_function.batch_to_gpu(batch)
        target = y[0].permute(0, 2, 1)  # y[0] is mel
        target = target[:,:-2,:]
        gate = y[1]  # y[1] is gate padding


        #target_fill_zero = torch.zeros(1, 2000 - target.size(1), target.size(2))
        #target_fill_zero = to_gpu(target_fill_zero).float()
        #target = torch.cat([target, target_fill_zero], dim=1)

        #gate_fill_one = torch.zeros(1, 2000 - gate.size(1))
        #gate_fill_one = to_gpu(gate_fill_one).float()
        #gate = torch.cat([gate, gate_fill_one], dim=1)

        mel_length = to_gpu(torch.tensor([target.size(1)]))
        pred_y_mel, pred_y_gate_output, predict_velocity, predict_pressure, predict_matrixA, predict_matrixB = model(x[0],mel_length)

        loss = criterion(pred_y_mel, pred_y_gate_output, target.float(), gate.float())
        print("Test loss", loss.item())

        #print(target)
        #print(pred_y_mel)
        print_spectrogram(pred_y_mel,pred_y_gate_output)
        print_spectrogram(target,gate,ground_truth=True)
        print_spectrogram(predict_velocity, gate, pic_name="g_velocity")
        print_spectrogram(predict_pressure, gate, pic_name="g_pressure")
        print_spectrogram(predict_matrixA, gate, pic_name="matrix_A")
        print_spectrogram(predict_matrixB, gate, pic_name="matrix_B")
        print_spectrogram(torch.square(pred_y_mel - target), gate, pic_name="error_spectrogram")

        print("Matrix A",predict_matrixA.size())
        matrix_A = predict_matrixA[:1,:1,:].squeeze().to("cpu").detach().numpy()
        matrix_B = predict_matrixB[:1,:1,:].squeeze().to("cpu").detach().numpy()

        g_v = predict_velocity[:1,:1,:].squeeze().to("cpu").detach().numpy()
        g_p = predict_pressure[:1,:1,:].squeeze().to("cpu").detach().numpy()
        print_fre(matrix_A,pic_name="v5_matrixA")
        print_fre(matrix_B, pic_name="v5_matrixB")
        print_fre(g_v, pic_name="v5_g_v")
        print_fre(g_p, pic_name="v5_g_p")
        break


tt_dataset()

