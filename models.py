import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

### Sin activation function
class Sin(nn.Module):
  def __init__(self):
    super(Sin, self).__init__()
    pass
  def forward(self, x):
    return torch.sin(x)
def cuda2numpy(a):
  return a.data.cpu().numpy()



### Simple MLP
class MLP(nn.Module):
  def __init__(self,architecture=[784,512,512,10],activation='sin'):
        super(MLP, self).__init__()
        if activation=='sin':
            self.activation=Sin()
        elif activation=='relu':
            self.activation=nn.ReLU()
        elif activation=='leakyrelu':
            self.activation=nn.LeakyReLU()
        elif activation=='tanh':
            self.activation=nn.Tanh()
        elif activation=='sigmoid':
            self.activation=nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function: {}".format(activation)) 
        self.architecture=architecture
        arch=[]
        for i in range(1,len(architecture)-1):
            arch.append(nn.Linear(architecture[i-1],architecture[i]))
            arch.append(self.activation)
        self.basis=nn.Sequential(*arch)
        self.regressor=nn.Linear(architecture[-2],architecture[-1])

  def forward(self,x):
        assert x.shape[1]==self.architecture[0]
        z=self.basis(x)
        out=self.regressor(z)
        return out

### Simple MLP with skip connections
# This MLP allows skip connections between layers, which can help with gradient flow and model performance
class MLPWithSkipConnections(nn.Module):
    def __init__(self, architecture=[784,512,512,10], activation='sin'):
        super(MLPWithSkipConnections, self).__init__()
        if activation=='sin':
            self.activation=Sin()
        elif activation=='relu':
            self.activation=nn.ReLU()
        elif activation=='leakyrelu':
            self.activation=nn.LeakyReLU()
        elif activation=='tanh':
            self.activation=nn.Tanh()
        elif activation=='sigmoid':
            self.activation=nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function: {}".format(activation))        
        self.architecture = architecture

        # Create individual layers instead of sequential
        self.layers = nn.ModuleList()
        for i in range(1, len(architecture)-1):
            self.layers.append(nn.Linear(architecture[i-1], architecture[i]))

        # Final regression layer
        self.regressor = nn.Linear(architecture[-2], architecture[-1])

        # Projection layers for skip connections when dimensions don't match
        self.skip_projections = nn.ModuleList()
        for i in range(1, len(architecture)-1):
            if architecture[i-1] != architecture[i]:
                # Need projection when input and output dimensions differ
                self.skip_projections.append(nn.Linear(architecture[i-1], architecture[i]))
            else:
                # Identity connection when dimensions match
                self.skip_projections.append(nn.Identity())

    def forward(self, x):
        assert x.shape[1] == self.architecture[0]

        current = x

        # Forward through hidden layers with skip connections
        for i, (layer, skip_proj) in enumerate(zip(self.layers, self.skip_projections)):
            # Store input for skip connection
            residual = current

            # Forward pass through current layer
            current = layer(current)
            current = self.activation(current)

            # Add skip connection (with projection if needed)
            current = current + skip_proj(residual)

        # Final regression layer (no skip connection here)
        out = self.regressor(current)
        return out
    

class MLPWithSkipConnectionsBN(nn.Module):
    def __init__(self, architecture=[784,512,512,10], activation='sin'):
        super(MLPWithSkipConnectionsBN, self).__init__()
        if activation == 'sin':
            self.activation = Sin()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        self.architecture = architecture
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.skip_projections = nn.ModuleList()

        for i in range(1, len(architecture) - 1):
            in_dim = architecture[i - 1]
            out_dim = architecture[i]

            self.layers.append(nn.Linear(in_dim, out_dim))
            self.batch_norms.append(nn.BatchNorm1d(out_dim))

            if in_dim != out_dim:
                self.skip_projections.append(nn.Linear(in_dim, out_dim))
            else:
                self.skip_projections.append(nn.Identity())

        self.regressor = nn.Linear(architecture[-2], architecture[-1])

    def forward(self, x):
        assert x.shape[1] == self.architecture[0]
        current = x

        for layer, bn, skip_proj in zip(self.layers, self.batch_norms, self.skip_projections):
            residual = current
            current = layer(current)
            current = bn(current)
            current = self.activation(current)
            current = current + skip_proj(residual)

        out = self.regressor(current)
        return out
    




class MLPwithDropout(nn.Module):
    def __init__(self, architecture=[784,512,512,10], activation='sin', dropout=0.0):
        super(MLPwithDropout, self).__init__()
        
        # Select activation function
        if activation=='sin':
            self.activation = Sin()  
        elif activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation=='tanh':
            self.activation = nn.Tanh()
        elif activation=='sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}") 
        
        self.architecture = architecture
        self.dropout = dropout
        
        layers = []
        num_layers = len(architecture) - 1
        
        # Only apply dropout to hidden layers (not input, not output)
        for i in range(1, num_layers):
            layers.append(nn.Linear(architecture[i-1], architecture[i]))
            layers.append(self.activation)
            if i < num_layers - 1 and dropout > 0.0:  # hidden layers only
                layers.append(nn.Dropout(p=dropout))
        
        self.basis = nn.Sequential(*layers)
        self.regressor = nn.Linear(architecture[-2], architecture[-1])

    def forward(self, x):
        assert x.shape[1] == self.architecture[0]
        z = self.basis(x)
        out = self.regressor(z)
        return out


import math


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

    
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

    
class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

    
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

    
class ResBlock(nn.Module):
    def __init__(self, layer, *args, **kwargs):
        super().__init__()
        if layer == 'Linear':
            self.block = nn.Sequential(
                nn.Linear(*args),
                kwargs['activation'],
                nn.Linear(*args)
            )
        elif layer == 'SAB':
            self.block = nn.Sequential(
                SAB(*args, **kwargs),
                SAB(*args, **kwargs),
            )
        elif layer == 'ISAB':
            self.block = nn.Sequential(
                ISAB(*args, **kwargs),
                ISAB(*args, **kwargs),
            )
        else:
            print(f'residual block for {layer} is not implemented')

    def forward(self, x):
        identity = x
        return identity + self.block(x)
    
    
    

from copy import deepcopy


class SetTransformer(nn.Module):
    def __init__(self, model_config):
        # num_outputs = model_config['num_output']
        dim_input = model_config['dim_input']
        dim_output = model_config['dim_output']
        num_inds = model_config['num_inds']
        dim_hidden = model_config['dim_hidden']
        num_heads = model_config['num_heads']
        ln = model_config['ln']
        params_budget = model_config['params_budget']

        super(SetTransformer, self).__init__()
        if params_budget is None:
            self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
            )
            self.dec = nn.Sequential(
                # PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            )
        else:
            enc_base = nn.ModuleList([
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                # ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
                ResBlock('ISAB', dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
            ])
            insert_id = 2
            enc = update_arch_under_budget(enc_base, insert_id, params_budget // 2, enc_base[1])
            self.enc = nn.Sequential(*enc)

            dec_base = nn.ModuleList([
                # PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                ResBlock('SAB', dim_hidden, dim_hidden, num_heads, ln=ln)
            ])
            insert_id = 2
            dec = update_arch_under_budget(dec_base, insert_id, params_budget // 2, dec_base[1])
            self.dec = nn.Sequential(*dec)

        self.output_mapper = nn.Linear(dim_hidden, dim_output)

    def forward(self, X):
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
            X = X.unsqueeze(-1)
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        assert len(X.shape) == 3, f'input shape {X.shape} should be 1 x n_elements x input_dim'
        enc = self.enc(X)
        dec = self.dec(enc)
        out = self.output_mapper(dec)
        return out.squeeze()


class DeepSet(nn.Module):
    def __init__(self, model_config):
        super(DeepSet, self).__init__()
        self.dim_hidden = model_config['dim_hidden']

        self.pooling = GAP()

        self.activation = nn.ReLU()
        self.dim_input = model_config['dim_input']
        self.dim_output = model_config['dim_output']
        self.num_output = model_config['num_output']
        self.params_budget = model_config['params_budget']

        if self.pool_name == 'SWE':
            first_dec_layer = nn.Linear(self.num_ref * self.num_slice, self.dim_hidden)
        elif self.pool_name == 'GAP':
            first_dec_layer = nn.Linear(self.dim_hidden, self.dim_hidden)

        if self.params_budget is None:
            self.enc = nn.Sequential(
                nn.Linear(self.dim_input, self.dim_hidden),
                self.activation,
                ResBlock('Linear', self.dim_hidden, self.dim_hidden, activation=nn.ReLU()),
                self.activation,
                nn.Linear(self.dim_hidden, self.dim_hidden))

            self.dec = nn.Sequential(
                first_dec_layer,
                self.activation,
                ResBlock('Linear', self.dim_hidden, self.dim_hidden, activation=nn.ReLU()),
                self.activation,
                nn.Linear(self.dim_hidden, self.num_output * self.dim_output))
        else:
            enc_base = nn.ModuleList([
                nn.Linear(self.dim_input, self.dim_hidden),
                self.activation,
                self.activation,
                nn.Linear(self.dim_hidden, self.dim_hidden)
            ])
            insert_id = 2
            block = ResBlock('Linear', self.dim_hidden, self.dim_hidden, activation=nn.ReLU())
            enc = update_arch_under_budget(enc_base, insert_id, self.params_budget // 2, block)
            self.enc = nn.Sequential(*enc)

            dec_base = nn.ModuleList([
                first_dec_layer,
                self.activation,
                self.activation,
                nn.Linear(self.dim_hidden, self.num_output * self.dim_output)
            ])
            insert_id = 2
            dec = update_arch_under_budget(dec_base, insert_id, self.params_budget // 2, block)
            self.dec = nn.Sequential(*dec)

    def forward(self, X):
        X = self.enc(X)
        # print(X.shape)
        X = self.pooling(X).reshape(1, -1)
        # print(X.shape)
        X = self.dec(X).reshape(self.num_output, self.dim_output)
        # print(X.shape, '\n')
        return X.squeeze()


def update_arch_under_budget(cur_arch, insert_id, budget, block):
    cur_num_params = 0
    for comp in cur_arch:
        if isinstance(comp, nn.Module):
            for _, param in comp.named_parameters():
                if param.requires_grad:
                    cur_num_params += param.numel()
    rem_budget = budget - cur_num_params

    block_params = 0
    for _, param in block.named_parameters():
        if param.requires_grad:
            block_params += param.numel()
    num_blocks = int(rem_budget // block_params)
    new_arch = cur_arch
    if num_blocks > 0:
        for _ in range(num_blocks):
            new_arch.insert(insert_id, deepcopy(block))
    return new_arch


class GAP(torch.nn.Module):
    def __init__(self):
        super(GAP, self).__init__()

    def forward(self, X):
        return X.mean(0).unsqueeze(0)
    




class pairSetTransformer(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        dim_input = model_config['dim_input']
        dim_hidden = model_config['dim_hidden']
        num_heads = model_config['num_heads']
        num_inds = model_config['num_inds']
        ln = model_config.get('ln', True)

        # Encode each point cloud independently
        self.enc_X = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        self.enc_Y = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )

        # Cross-attention between sets (X attends to Y, and Y attends to X)
        self.cross_X = MAB(dim_hidden, dim_hidden, dim_hidden, num_heads, ln=ln)
        self.cross_Y = MAB(dim_hidden, dim_hidden, dim_hidden, num_heads, ln=ln)

        # Decoder to produce per-point scalar responses
        self.dec_X = nn.Sequential(
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, 1)
        )
        self.dec_Y = nn.Sequential(
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, 1)
        )

    def forward(self, X, Y):
        """
        X: [B, N, d]
        Y: [B, M, d]
        Returns:
          RX: [B, N, 1]
          RY: [B, M, 1]
        """
        # Encode independently
        fX = self.enc_X(X)  # [B, N, h]
        fY = self.enc_Y(Y)  # [B, M, h]

        # Cross-attention: each point in X attends to Y, and vice versa
        fX_cross = self.cross_X(fX, fY)
        fY_cross = self.cross_Y(fY, fX)

        # Decode to scalar scores
        RX = self.dec_X(fX_cross).squeeze(-1)  # [B, N, 1]
        RY = self.dec_Y(fY_cross).squeeze(-1)  # [B, M, 1]

        return RX, RY
