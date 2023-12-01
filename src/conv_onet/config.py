from src.conv_onet import models


def get_model(cfg):
    """
    Return the network model.

    Args:
        cfg (dict): imported yaml config.

    Returns:
        decoder (nn.module): the network model.
    """

    dim = cfg['data']['dim']
    low_grid_len = cfg['grid_len']['low']
    high_grid_len = cfg['grid_len']['high']
    color_grid_len = cfg['grid_len']['color']
    c_dim = cfg['model']['c_dim']  # feature dimensions
    pos_embedding_method = cfg['model']['pos_embedding_method']

    decoder = models.decoder_dict['dfprior'](
        dim=dim, c_dim=c_dim, 
        low_grid_len=low_grid_len, high_grid_len=high_grid_len,
        color_grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)
 
    return decoder
