from types import SimpleNamespace


def dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dictionary and its nested dictionnaries to a SimpleNamespace.
    """
    content = {}
    for k, v in d.items():
        content[k] = dict_to_namespace(v) if isinstance(v, dict) else v

    return SimpleNamespace(**content)


def config_is_valid(config: SimpleNamespace) -> bool:
    """Check if the config is consistent.
    """
    # TODO: if something is invalid, raise more explicit exception instead of just returning False
    return all([
        # Check autoencoder dimensions
        config.autoencoder.encoder_dimensions[0] == 3,
        config.autoencoder.encoder_dimensions[-1] == config.gfv_dim,
        config.autoencoder.decoder_dimensions[0] == config.gfv_dim,
        config.autoencoder.decoder_dimensions[-1] == 3,

        # Check GAN dimensions
        config.gan.generator_dimensions[0] == config.z_dim,
        config.gan.generator_dimensions[-1] == config.gfv_dim,
        config.gan.critic_dimensions[0] == config.gfv_dim,
        config.gan.critic_dimensions[-1] == 1,

        # Check dataset
        config.dataset in ("dental", "shapenet"),
    ])
