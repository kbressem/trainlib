import munch
from monai.networks.nets import UNet
import munch

def get_model(config: munch.Munch):
    """Create a standard UNet"""
    return UNet(
        spatial_dims=config.ndim,
        in_channels=len(config.data.image_cols),
        out_channels=config.model.out_channels,
        channels=config.model.channels,
        strides=config.model.strides,
        num_res_units=config.model.num_res_units,
        act=config.model.act,
        norm=config.model.norm,
        dropout=config.model.dropout,
    )
