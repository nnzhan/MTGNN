import torch.optim as optim
from model import *
import util
class trainer():
    def __init__(self, scaler, in_dim, out_dim, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn, addaptadj, node_dim, k ,gdep, gx=None):
        #self.model = gxnet(num_nodes, device, supports, aptinit, addaptadj, gcn_bool)

        self.model = gxnet(num_nodes, device, supports, addaptadj, gcn, gdep, gx=gx, k=k, node_dim=node_dim, dropout=dropout, conv_channels=nhid*3, residual_channels=nhid*3, skip_channels=nhid*4, end_channels=nhid*8, out_dim=out_dim)
        self.model.to(device)
        # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lrate, weight_decay=wdecay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)

        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.step = 10000
        self.iter = 0
        self.m = 1
    def train(self, input, real_val, idx=None):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, idx=idx)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        # if self.iter%4==2:
        #     loss = self.loss(predict[:,:,:,:8], real[:,:,:,:8], 0.0)
        # elif self.iter%4==3:
        #     loss = self.loss(predict, real, 0.0)
        # else:
        #     loss = self.loss(predict[:,:,:,:4], real[:,:,:,:4], 0.0)
        if self.iter%2700==0 and self.m<=12:
            self.m +=1
        loss = self.loss(predict[:, :, :, :self.m], real[:, :, :, :self.m], 0.0)
        # if self.iter>2*self.step:
        #     loss = self.loss(predict, real, 0.0)
        # elif self.iter>self.step:
        #     loss = self.loss(predict[:, :, :, :8], real[:, :, :, :8], 0.0)
        # else:
        #     loss = self.loss(predict[:, :, :, :4], real[:, :, :, :4], 0.0)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self.clip)

        self.optimizer.step()
        # mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        self.iter += 1
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        # real = real[:,:,:,6:]
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
