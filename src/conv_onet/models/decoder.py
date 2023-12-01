import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)


class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.

    """

    def __init__(self, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class Same(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze(0)
        return x


class MLP(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from low level and concat to the current feature.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear',
                 color=False, skips=[2], grid_len=0.16, pos_embedding_method='fourier', concat_feature=False):
        super().__init__()
        self.name = name
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf':
            if 'color' in name:
                multires = 10
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=False)
            embedding_size = multires*6+3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])

        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True,
                          mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_grid=None):
        if self.c_dim != 0:
            c = self.sample_grid_feature(
                p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)

            if self.concat_feature:
                # only happen to high decoder, get feature from low level and concat to the current feature
                with torch.no_grad():
                    c_low = self.sample_grid_feature(
                        p, c_grid['grid_low']).transpose(1, 2).squeeze(0)
                c = torch.cat([c, c_low], dim=1)

        p = p.float()

        embedded_pts = self.embedder(p)
        h = embedded_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if self.c_dim != 0:
                h = h + self.fc_c[i](c)
            if i in self.skips:
                h = torch.cat([embedded_pts, h], -1)
        out = self.output_linear(h)
        if not self.color:
            out = out.squeeze(-1)
        return out


class mlp_tsdf(nn.Module):
    """
    Attention-based MLP.

    """

    def __init__(self):
        super().__init__()

        self.no_grad_feature = False
        self.sample_mode = 'bilinear'

        self.pts_linears = nn.ModuleList(
            [DenseLayer(2, 64, activation="relu")] +
            [DenseLayer(64, 128, activation="relu")] +
            [DenseLayer(128, 128, activation="relu")] +
            [DenseLayer(128, 64, activation="relu")])

        self.output_linear = DenseLayer(
                64, 2, activation="linear") #linear

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid() 

    def sample_grid_tsdf(self, p, tsdf_volume, tsdf_bnds, device='cuda:0'):
        p_nor = normalize_3d_coordinate(p.clone(), tsdf_bnds)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
            # acutally trilinear interpolation if mode = 'bilinear'
        tsdf_value = F.grid_sample(tsdf_volume.to(device), vgrid.to(device), padding_mode='border', align_corners=True,
                            mode='bilinear').squeeze(-1).squeeze(-1)
        
        return tsdf_value

    def forward(self, p, occ, tsdf_volume, tsdf_bnds, **kwargs):
        tsdf_val = self.sample_grid_tsdf(p, tsdf_volume, tsdf_bnds, device='cuda:0')
        tsdf_val = tsdf_val.squeeze(0)
        
        tsdf_val = 1. - (tsdf_val + 1.) / 2. #0,1
        tsdf_val = torch.clamp(tsdf_val, 0.0, 1.0)
        occ = occ.reshape(tsdf_val.shape)
        inv_tsdf = -0.1 * torch.log((1 / (tsdf_val + 1e-8)) - 1 + 1e-7) #0.1
        inv_tsdf = torch.clamp(inv_tsdf, -100.0, 100.0)
        input = torch.cat([occ, inv_tsdf], dim=0)
        h = input.t()
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        weight = self.output_linear(h)
        weight = self.softmax(weight)
        out = weight.mul(input.t()).sum(dim=1)

        return out, weight[:, 1]
        


class DF(nn.Module):
    """    
    Neural Implicit Scalable Encoding.

    Args:
        dim (int): input dimension.
        c_dim (int): feature dimension.
        low_grid_len (float): voxel length in low grid.
        high_grid_len (float): voxel length in high grid.
        color_grid_len (float): voxel length in color grid.
        hidden_size (int): hidden size of decoder network
        pos_embedding_method (str): positional embedding method.
    """

    def __init__(self, dim=3, c_dim=32,
                 low_grid_len=0.16, high_grid_len=0.16,
                 color_grid_len=0.16, hidden_size=32, pos_embedding_method='fourier'):
        super().__init__()

       
        self.low_decoder = MLP(name='low', dim=dim, c_dim=c_dim, color=False,
                                  skips=[2], n_blocks=5, hidden_size=hidden_size,
                                  grid_len=low_grid_len, pos_embedding_method=pos_embedding_method)
        self.high_decoder = MLP(name='high', dim=dim, c_dim=c_dim*2, color=False,
                                skips=[2], n_blocks=5, hidden_size=hidden_size,
                                grid_len=high_grid_len, concat_feature=True, pos_embedding_method=pos_embedding_method)
        self.color_decoder = MLP(name='color', dim=dim, c_dim=c_dim, color=True,
                                 skips=[2], n_blocks=5, hidden_size=hidden_size,
                                 grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)

        self.mlp = mlp_tsdf()


    def sample_grid_tsdf(self, p, tsdf_volume, tsdf_bnds, device='cuda:0'):

        p_nor = normalize_3d_coordinate(p.clone(), tsdf_bnds)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
            # acutally trilinear interpolation if mode = 'bilinear'
        tsdf_value = F.grid_sample(tsdf_volume.to(device), vgrid.to(device), padding_mode='border', align_corners=True,
                            mode='bilinear').squeeze(-1).squeeze(-1)
        return tsdf_value



    def forward(self, p, c_grid, tsdf_volume, tsdf_bnds, stage='low', **kwargs):
        """
            Output occupancy/color in different stage.
        """
      
        device = f'cuda:{p.get_device()}'
        if stage == 'low':
            low_occ = self.low_decoder(p, c_grid)
            low_occ = low_occ.squeeze(0)

            w = torch.ones(low_occ.shape[0]).to(device)
            raw = torch.zeros(low_occ.shape[0], 4).to(device).float()
            raw[..., -1] = low_occ # new_occ
            return raw, w
        elif stage == 'high':
            high_occ = self.high_decoder(p, c_grid)
            raw = torch.zeros(high_occ.shape[0], 4).to(device).float()
            low_occ = self.low_decoder(p, c_grid)
            low_occ = low_occ.squeeze(0)
            f_add_m_occ = high_occ + low_occ

            eval_tsdf = self.sample_grid_tsdf(p, tsdf_volume, tsdf_bnds, device)
            eval_tsdf_mask = ((eval_tsdf > -1.0+1e-4) & (eval_tsdf < 1.0-1e-4))
            eval_tsdf_mask = eval_tsdf_mask.squeeze()
            
            w = torch.ones(low_occ.shape[0]).to(device)
            low_occ[eval_tsdf_mask], w[eval_tsdf_mask] = self.mlp(p[:, eval_tsdf_mask, :], f_add_m_occ[eval_tsdf_mask], tsdf_volume, tsdf_bnds)
            new_occ = low_occ
            new_occ = new_occ.squeeze(-1)
            raw[..., -1] = new_occ
            return raw, w 
        elif stage == 'color':
            high_occ = self.high_decoder(p, c_grid)
            raw = self.color_decoder(p, c_grid)
            low_occ = self.low_decoder(p, c_grid)
            low_occ = low_occ.squeeze(0)
            f_add_m_occ = high_occ + low_occ

            eval_tsdf = self.sample_grid_tsdf(p, tsdf_volume, tsdf_bnds, device)
            eval_tsdf_mask = ((eval_tsdf > -1.0+1e-4) & (eval_tsdf < 1.0-1e-4))
            eval_tsdf_mask = eval_tsdf_mask.squeeze()
            
            w = torch.ones(low_occ.shape[0]).to(device)
            low_occ[eval_tsdf_mask], w[eval_tsdf_mask] = self.mlp(p[:, eval_tsdf_mask, :], f_add_m_occ[eval_tsdf_mask], tsdf_volume, tsdf_bnds)
            new_occ = low_occ
            raw[..., -1] = new_occ          
            return raw, w
    