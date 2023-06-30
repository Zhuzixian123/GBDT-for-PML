import torch
import torch.nn as nn

class Line(nn.Module):
    def __init__(self,in_features,out_features,config):
        super(Line,self).__init__()
        self.config=config
        self.first_output=config.model_params['first_output']
        self.hidden_size = config.model_params['hidden_size']
        self.last_output = config.model_params['last_output']

        self.line1=nn.Linear(in_features,self.first_output)
        self.line2 = nn.Linear(self.first_output, self.hidden_size)
        self.line3 = nn.Linear(self.hidden_size,self.hidden_size)
        self.line4 = nn.Linear(self.hidden_size,self.hidden_size)
        self.line5 = nn.Linear(self.hidden_size,self.hidden_size)
        self.line6 = nn.Linear(self.hidden_size,self.last_output)
        self.line7 = nn.Linear(self.last_output, out_features)

        self.loss=config.train_params['loss_fc']

    def forward(self,x,y):
        x = self.line1(x)
        x=torch.tanh(x)
        x = self.line2(x)
        x=torch.tanh(x)
        x = self.line3(x)
        x=torch.relu(x)
        x = self.line4(x)
        x=torch.relu(x)
        x = self.line5(x)
        x=torch.tanh(x)
        x = self.line6(x)
        x=torch.tanh(x)
        y_pre = self.line7(x)

        #计算loss
        loss=self.loss_fc(y_pre,y)

        return y_pre,loss

    def loss_fc(self,y_pre,y):

        loss=self.loss(y_pre,y)

        return loss
