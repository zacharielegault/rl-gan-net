from types import SimpleNamespace


def dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dictionary and its nested dictionnaries to a SimpleNamespace.
    """
    content = {}
    for k, v in d.items():
        content[k] = dict_to_namespace(v) if isinstance(v, dict) else v

    return SimpleNamespace(**content)


def config_is_valid(config: SimpleNamespace):
    """Check if the config is consistent. Raises ValueError if some values are inconsistent.
    """
    # Check autoencoder dimensions
    if not config.autoencoder.encoder_dimensions[0] == 3:
        raise ValueError(f"Encoder's first dimension should be 3, got {config.autoencoder.encoder_dimensions[0]}.")
    if not config.autoencoder.encoder_dimensions[-1] == config.gfv_dim:
        raise ValueError(f"Encoder's last dimension should be the same as GFV dimension, got "
                         f"{config.autoencoder.encoder_dimensions[-1]} and {config.gfv_dim}.")
    if not config.autoencoder.decoder_dimensions[0] == config.gfv_dim:
        raise ValueError(f"Decoder's first dimension should be the same as GFV dimension, got "
                         f"{config.autoencoder.decoder_dimensions[0]} and {config.gfv_dim}.")
    if not config.autoencoder.decoder_dimensions[-1] == 3:
        raise ValueError(f"Decoder's last dimension should be 3, got {config.autoencoder.decoder_dimensions[-1]}.")

    # Check GAN dimensions
    if not config.gan.generator_dimensions[0] == config.z_dim:
        raise ValueError(f"Generator's first dimension should be the same as Z dimension, got "
                         f"{config.gan.generator_dimensions[0]} and {config.z_dim}.")
    if not config.gan.generator_dimensions[-1] == config.gfv_dim:
        raise ValueError(f"Generator's last dimension should be the same as GFV dimension, got "
                         f"{config.gan.generator_dimensions[-1]} and {config.gfv_dim}.")
    if not config.gan.critic_dimensions[0] == config.gfv_dim:
        raise ValueError(f"Critic's first dimension should be the same as GFV dimension, got "
                         f"{config.gan.critic_dimensions[0]} and {config.gfv_dim}.")
    if not config.gan.critic_dimensions[-1] == 1:
        raise ValueError(f"Critic's last dimension should be 1, got {config.gan.critic_dimensions[-1]}.")

    # Check dataset
    if not config.dataset in ("dental", "shapenet"):
        raise ValueError(f"Dataset should be one of ['dental', 'shapenet'], got {config.dataset}.")

    if config.dataset == "shapenet" and config.num_points > 2048:
        raise ValueError(f"With ShapeNet dataset, maximum num_points is 2048, got {config.num_points}.")
