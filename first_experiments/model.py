import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.layers = nn.ModuleDict()

        dims = [config['input_dim']] + config['fc'] + [config['output_dim']]
        for i in range(len(dims)-1):
            self.layers[f'fc{i+1}'] = nn.Linear(dims[i], dims[i+1])
            if i < len(dims) - 2:
                self.layers[f'relu{i+1}'] = nn.ReLU()
        self.net = nn.Sequential(*self.layers.values())
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # <-- FLATTEN
        return self.net(x)
    

class CNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleDict()

        input_dim = config['input_dim']      # (H, W)
        output_dim = config['output_dim']
        conv_dims = config['conv']
        linear_dims = config['fc']

        # --------------------
        # Convolutional layers
        # --------------------
        conv_channels = [config.get('in_channels', 1)] + conv_dims  # grayscale MNIST

        for i in range(len(conv_channels) - 1):
            self.layers[f'conv{i+1}'] = nn.Conv2d(
                conv_channels[i],
                conv_channels[i+1],
                kernel_size=config['conv_kernel'][i],
                stride=config['conv_stride'][i],
                padding=config['conv_padding'][i]
            )

            self.layers[f'relu{i+1}'] = nn.ReLU()

            if i < len(config['max_pool_kernel']) and config['max_pool_kernel'][i] is not None:
                self.layers[f'pool{i+1}'] = nn.MaxPool2d(
                    kernel_size=config['max_pool_kernel'][i],
                    stride=config['max_pool_stride'][i]
                )

        # --------------------
        # Infer flattened dim (conv layers ONLY)
        # --------------------
        with torch.no_grad():
            x = torch.zeros(1, config.get('in_channels', 1), *input_dim)
            for name, layer in self.layers.items():
                if name.startswith(('conv', 'relu', 'pool')):
                    x = layer(x)
            flat_dim = x.view(1, -1).size(1)

        self.layers['flatten'] = nn.Flatten()

        # --------------------
        # Fully connected layers
        # --------------------
        linear = [flat_dim] + linear_dims + [output_dim]

        for i in range(len(linear) - 1):
            self.layers[f'fc{i+1}'] = nn.Linear(linear[i], linear[i+1])

            if i < len(linear) - 2:
                self.layers[f'relu_fc{i+1}'] = nn.ReLU()

        self.net = nn.Sequential(*self.layers.values())


    def forward(self, x):
        return self.net(x)


import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Bloco residual básico: duas convoluções com shortcut connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Shortcut: ajusta dimensões quando canal ou resolução mudam
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # conexão residual
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleDict()

        input_dim    = config['input_dim']           # (H, W)
        output_dim   = config['output_dim']
        res_channels = config['res_channels']        # ex: [16, 32, 64]
        blocks_per_stage = config.get('blocks_per_stage', [2, 2, 2])
        linear_dims  = config['fc']
        in_ch        = config.get('in_channels', 1)

        # --------------------
        # Stem: conv inicial
        # --------------------
        stem_ch = config.get('stem_channels', res_channels[0])
        self.layers['stem_conv'] = nn.Conv2d(in_ch, stem_ch,
                                             kernel_size=3, stride=1,
                                             padding=1, bias=False)
        self.layers['stem_bn']   = nn.BatchNorm2d(stem_ch)
        self.layers['stem_relu'] = nn.ReLU(inplace=True)

        # --------------------
        # Estágios residuais
        # --------------------
        current_ch = stem_ch
        for stage_idx, (out_ch, n_blocks) in enumerate(
                zip(res_channels, blocks_per_stage)):

            stride = 1 if stage_idx == 0 else 2   # downsampling a partir do 2º estágio

            for block_idx in range(n_blocks):
                block_stride = stride if block_idx == 0 else 1
                name = f'res{stage_idx+1}_{block_idx+1}'
                self.layers[name] = ResidualBlock(current_ch, out_ch, block_stride)
                current_ch = out_ch

        # --------------------
        # Pooling global + flatten
        # --------------------
        self.layers['global_pool'] = nn.AdaptiveAvgPool2d((1, 1))
        self.layers['flatten']     = nn.Flatten()

        # --------------------
        # Cabeça fully connected
        # --------------------
        fc_dims = [current_ch] + linear_dims + [output_dim]
        for i in range(len(fc_dims) - 1):
            self.layers[f'fc{i+1}'] = nn.Linear(fc_dims[i], fc_dims[i+1])
            if i < len(fc_dims) - 2:
                self.layers[f'relu_fc{i+1}'] = nn.ReLU(inplace=True)

        self.net = nn.Sequential(*self.layers.values())

    def forward(self, x):
        return self.net(x)