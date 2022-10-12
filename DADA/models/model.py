# Following the modeles used in:
#  https://github.com/hfawaz/dl-4-tsc/tree/master/classifiers
# and covert the code from Tensorflow-based to Pytorch-based.

import torch
import torch.nn as nn
import torch.nn.functional as F



def get_model(model_name, input_length, num_classes):
    if model_name == 'model_mlp':
        model = model_mlp(input_length, num_classes)
    elif model_name == 'model_conv1d':
        model = model_conv1d(input_length, num_classes=num_classes)
    elif model_name == 'model_ResNet':
        model = model_ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    else:
        raise NameError('no model named, %s' % model_name)
    return model



class model_lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes = 2, num_layers = 2, dropout_rate=0.6):
        super(model_lstm, self).__init__()
        self.input_dim = input_dim;
        self.hidden_dim = hidden_dim;
        self.dropout_rate = dropout_rate;
        self.num_layers = num_layers;  # number of recurrent layers, here 2 means stacking two LSTMs.
        self.num_classes = num_classes;

        # defining modules.
        # 
        self.lstm = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim,  num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)


    def forward(self, a, b):
        batch_size = a.size(0)
        x = torch.cat((a,b), axis=2)
        output, (h_n, c_n) = self.lstm(x)
        logits = self.fc(output[:, -1, :])
        return logits 



class model_mlp(nn.Module):
    def __init__(self, input_length, num_classes, dropout_rate=0.5, is_training=True):
        super(model_mlp,self).__init__()
        self.input_length = input_length;
        self.num_classes = num_classes;
        self.is_training = is_training;

        # define modules.
        self.fc1 = nn.Linear(self.input_length, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, self.num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x is expected with the shape of [Batch_size, input_length]
        x = torch.squeeze(x)    # from [Batch_size, input_length, 1] => [Batch_size, input_length]
        output =  self.dropout(self.relu(self.bn1(self.fc1(x))))
        output =  self.dropout(self.relu(self.bn2(self.fc2(output))))
        output =  self.fc3(output)
        return output

    
class model_conv1d(nn.Module):
    def __init__(self, input_length, channel_size=1, num_classes=2, dropout_rate=0.5, is_training=True):
        super(model_conv1d, self).__init__()
        self.input_length = input_length;
        self.channel_size = channel_size;
        self.num_classes = num_classes;
        self.is_training = is_training;

        kernel_size = 5

        # input signal with size [N, C_in, L]
        # and output [N, C_out, L_out]
        self.conv1 = nn.Conv1d(in_channels=self.channel_size, out_channels=32, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn4 = nn.BatchNorm1d(256)
        self.avgpool = nn.AvgPool1d(27)  #TODO
        self.fc = nn.Linear(256, self.num_classes)

        self.pool = nn.MaxPool1d(3)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2) # dimensions of dim-1 and dim 2 are swappted. [B, Length, C] => [B, C, Length]
        batch_size = x.size(0)
        output = self.pool(self.relu( self.bn1(self.conv1(x))));
        output = self.pool(self.relu( self.bn2(self.conv2(output))));
        output = self.pool(self.relu( self.bn3(self.conv3(output))));
        output = self.relu(self.bn4(self.conv4(output)))
        #print(output.shape)
        output = self.avgpool(output)
        #print(output.shape)
        output = output.view(batch_size, -1)
        #print(output.shape)
        output = self.dropout(output)
        output = self.fc(output)


        return output

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x;

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class model_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, dropout_rate=0.5, is_training=True):
        super(model_ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.avgpool = nn.AvgPool1d(kernel_size=47)  # TODO
        self.fc = nn.Linear(512, num_classes)   # the value is undecided yet.
        self.dropout = nn.Dropout(dropout_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1):
        downsample = None;
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                    nn.Conv1d(self.inplanes, planes*block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(planes*block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride,  downsample=downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))
        
        return  nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2) # dimensions of dim-1 and dim 2 are swappted. [B, Length, C] => [B, C, Length]
        x = self.conv1(x);
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print(x.shape)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

# to define a resnet model
#  model = model_ResNet(BasicBlock, [2, 2, 2, 2])


