import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FC_Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC_Net, self).__init__()
        self.fc1 = nn.Linear(in_dim, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 120)
        self.fc4 = nn.Linear(120, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.sigmoid(x)
        return x


class ComNet1(nn.Module):
    """
    implementation of ComNet as in the paper:
    ComNet: Combination of Deep Learning and Expert Knowledge in OFDM Receivers
    ComNet1 denotes the 1st architecture where the SD net is MLP and takes [z_f, h_ls, y_d] as input
    """
    def __init__(self, CE_dim, SD_n):
        super(ComNet1, self).__init__()
        # CE_dim_in = CE_dim_out=128
        # SD_n = [input_dim, 120, 48]
        self.CE = nn.Linear(CE_dim, CE_dim)
        self.SD = nn.Sequential(
            nn.Linear(SD_n[0], SD_n[1]),
            nn.ReLU(),
            nn.Linear(SD_n[1], SD_n[2]),
            nn.Sigmoid()
        )
        #self.W = W # weight initialization for CE
        # # weight initialization with He initialization for SD net
        # with torch.no_grad():
        #     self.CE.weight = nn.Parameter(self.W)
        # nn.init.constant_(self.CE.bias, 0.0)
        for name, param in self.SD.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.kaiming_normal_(param)

    def forward(self, y_d, h_ls):
        #y_d, h_ls = x # yd is the received signal wit 8 symbols or 16 bits for QAM
        #y_d = x[:128]
        #h_ls = x[128:]
        h_tilde = self.CE(h_ls) # complex tensor
        batch_size = h_tilde.size()[0]
        h_tilde_c = torch.view_as_complex(h_tilde.view(-1, 2)) # real and img part
        y_c = torch.view_as_complex(y_d.view(-1, 2))
        x_zf = y_c/h_tilde_c
        x_zf = torch.view_as_real(x_zf)
        x_zf = x_zf.view(batch_size, -1)
        x_SD = torch.cat((y_d, h_tilde, x_zf), 1)
        b = self.SD(x_SD)
        return b


class ComNet2(nn.Module):
    """
    implementation of ComNet as in the paper:
    ComNet: Combination of Deep Learning and Expert Knowledge in OFDM Receivers
    ComNet2 denotes the 1st architecture where the SD net is MLP and takes z_f as input
    """
    def __init__(self, CE_dim, SD_n):
        super(ComNet2, self).__init__()
        # CE_dim_in = CE_dim_out=128
        # SD_n = [input_dim, 120, 48]
        self.CE = nn.Linear(CE_dim, CE_dim)
        self.SD = nn.Sequential(
            nn.Linear(SD_n[0], SD_n[1]),
            nn.ReLU(),
            nn.Linear(SD_n[1], SD_n[2]),
            nn.Sigmoid()
        )

        # weight initialization with He initialization for SD net
        for name, param in self.SD.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.kaiming_normal_(param)

    def forward(self, y_d, h_ls):
     # input is z_f
        h_tilde = self.CE(h_ls)  # complex tensor
        batch_size = h_tilde.size()[0]
        h_tilde_c = torch.view_as_complex(h_tilde.view(-1, 2))  # real and img part
        y_c = torch.view_as_complex(y_d.view(-1, 2))
        x_zf = y_c/h_tilde_c
        x_zf = torch.view_as_real(x_zf)
        x_zf = x_zf.view(batch_size, -1)
        #x_SD = torch.cat((y_d, h_tilde, x_zf), 1)
        b = self.SD(x_zf)
        return b


class ComNet3(nn.Module):
    """
    implementation of ComNet as in the paper:
    ComNet: Combination of Deep Learning and Expert Knowledge in OFDM Receivers
    ComNet2 denotes the 1st architecture where the SD net is BiLSTM and takes [yd, hs,zf] as input
    """
    def __init__(self, CE_dim, hidden_size, device):
        super(ComNet3, self).__init__()
        # CE_dim_in = CE_dim_out=128
        # SD_n = lstm parameters
        self.sequence_len = 64
        self.hidden_dim =hidden_size
        self.dropout = nn.Dropout(0.5)
        self.CE = nn.Linear(CE_dim, CE_dim)
        self.device = device

        self.SD1 = nn.LSTMCell(input_size=6, hidden_size=hidden_size[0])
        self.SD2 = nn.LSTMCell(input_size=hidden_size[0]*2, hidden_size=hidden_size[1])
        self.SD3 = nn.LSTMCell(input_size=hidden_size[1]*2, hidden_size=hidden_size[2])
        self.linear = nn.Sequential(
            nn.Linear(hidden_size[2], 48),
            nn.Sigmoid()
        )

        # weight initialization with He initialization for SD net
        # for name, param in self.SD.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     else:
        #         nn.init.kaiming_normal_(param)

    def forward(self, y_d, h_ls):
        # input is z_f
        h_tilde = self.CE(h_ls)  # complex tensor
        batch_size = h_tilde.size()[0]
        h_tilde_c = torch.view_as_complex(h_tilde.view(-1, 2))  # real and img part
        y_c = torch.view_as_complex(y_d.view(-1, 2))
        x_zf = y_c/h_tilde_c
        x_zf = torch.view_as_real(x_zf)
        x_zf = x_zf.view(batch_size, -1)
        x_SD = torch.cat((y_d, h_tilde, x_zf), 1) # x_SD: [batch, feature_dim]
        out = x_SD.view(self.sequence_len, batch_size, -1)

        # Creation of cell state and hidden state for layer 1
        hidden_state_layer_1 = torch.zeros(batch_size, self.hidden_dim[0], dtype=torch.float32, device=self.device)
        cell_state_layer_1 = torch.zeros(batch_size, self.hidden_dim[0], dtype=torch.float32, device=self.device)

        # Creation of cell state and hidden state for layer 2
        hidden_state_layer_2 = torch.zeros(batch_size, self.hidden_dim[1], dtype=torch.float32, device=self.device)
        cell_state_layer_2 = torch.zeros(batch_size, self.hidden_dim[1], dtype=torch.float32, device=self.device)

        # Creation of cell state and hidden state for layer 3
        hidden_state_layer_3 = torch.zeros(batch_size, self.hidden_dim[2], dtype=torch.float32, device=self.device)
        cell_state_layer_3 = torch.zeros(batch_size, self.hidden_dim[2], dtype=torch.float32, device=self.device)

        # Weights initialization
        torch.nn.init.kaiming_normal_(hidden_state_layer_1)
        torch.nn.init.kaiming_normal_(cell_state_layer_1)

        torch.nn.init.kaiming_normal_(hidden_state_layer_2)
        torch.nn.init.kaiming_normal_(cell_state_layer_2)

        torch.nn.init.kaiming_normal_(hidden_state_layer_2)
        torch.nn.init.kaiming_normal_(cell_state_layer_2)
        forward1, backward1 = [], []
        forward2, backward2 = [], []
        forward3, backward3 = [], []
        # forward pass for layer 1
        for i in range(self.sequence_len):
            hidden_state_layer_1, cell_state_layer_1 = self.SD1(out[i], (
                hidden_state_layer_1, cell_state_layer_1))
            hidden_state_layer_1 = self.dropout(hidden_state_layer_1)
            cell_state_layer_1 = self.dropout(cell_state_layer_1)
            forward1.append(hidden_state_layer_1)
        # reverse pass for layer 1
        for i in reversed(range(self.sequence_len)):
            hidden_state_layer_1, cell_state_layer_1 = self.SD1(out[i], (
                hidden_state_layer_1, cell_state_layer_1))
            hidden_state_layer_1 = self.dropout(hidden_state_layer_1)
            cell_state_layer_1 = self.dropout(cell_state_layer_1)
            backward1.append(hidden_state_layer_1)

        # forward pass for layer 2
        for fwd, back in zip(forward1, backward1):
            hidden_state_layer_2, cell_state_layer_2 = self.SD2(torch.cat((fwd, back), 1), (
                hidden_state_layer_2, cell_state_layer_2))
            hidden_state_layer_2 = self.dropout(hidden_state_layer_2)
            cell_state_layer_2 = self.dropout(cell_state_layer_2)
            forward2.append(hidden_state_layer_2)

        # reverse pass for layer 2
        for fwd, back in reversed(list(zip(forward1, backward1))):
            hidden_state_layer_2, cell_state_layer_2 = self.SD2(torch.cat((fwd, back), 1), (
                hidden_state_layer_2, cell_state_layer_2))
            hidden_state_layer_2 = self.dropout(hidden_state_layer_2)
            cell_state_layer_2 = self.dropout(cell_state_layer_2)
            backward2.append(hidden_state_layer_2)

            # forward pass for layer 3
        for fwd, back in zip(forward2, backward2):
            hidden_state_layer_3, cell_state_layer_3 = self.SD3(torch.cat((fwd, back), 1), (
                hidden_state_layer_3, cell_state_layer_3))
            hidden_state_layer_3 = self.dropout(hidden_state_layer_3)
            cell_state_layer_3 = self.dropout(cell_state_layer_3)
            forward3.append(hidden_state_layer_3)

        # reverse pass for layer 3
        for fwd, back in reversed(list(zip(forward2, backward2))):
            hidden_state_layer_3, cell_state_layer_3 = self.SD3(torch.cat((fwd, back), 1), (
                hidden_state_layer_3, cell_state_layer_3))
            hidden_state_layer_3 = self.dropout(hidden_state_layer_3)
            cell_state_layer_3 = self.dropout(cell_state_layer_3)
            backward3.append(hidden_state_layer_3)

        out = self.linear(hidden_state_layer_3)
        return out


class CENet(nn.Module):
    def __init__(self, CE_dim, W):
        super(CENet, self).__init__()
        self.CE = nn.Linear(CE_dim, CE_dim)
        self.W = W  # weight initialization for CE
        # weight initialization with He initialization for SD net
        with torch.no_grad():
            self.CE.weight = nn.Parameter(self.W)
        nn.init.constant_(self.CE.bias, 0.0)

    def forward(self, h_ls):
        return self.CE(h_ls)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

