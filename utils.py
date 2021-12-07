from types import SimpleNamespace


def dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dictionary and its nested dictionnaries to a SimpleNamespace.
    """
    content = {}
    for k, v in d.items():
        content[k] = dict_to_namespace(v) if isinstance(v, dict) else v

    return SimpleNamespace(**content)
