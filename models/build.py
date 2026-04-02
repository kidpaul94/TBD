from utils import registry

MODELS = registry.Registry('models')


def build_model_from_cfg(cfg, **kwargs):
    """
    Build a model from an EasyDict config (the ``model`` section of a YAML).

    Dispatch logic
    --------------
    1. Read ``cfg.NAME`` to identify the model class.
    2. If the class exposes ``build_from_cfg(cfg)``, call that.
       This classmethod is responsible for unpacking the nested YAML
       structure (e.g. ``mamba_config``) into the flat SimpleNamespace
       that ``PointScan.__init__`` expects.
    3. Fall back to ``MODELS.build(cfg)`` for classes that do not
       implement ``build_from_cfg`` (backward-compatible).

    Args:
        cfg:     EasyDict — the ``model`` sub-dict from cfg_from_yaml_file().
        **kwargs: Forwarded to ``build_from_cfg`` or ``MODELS.build``.

    Returns:
        Instantiated nn.Module.
    """
    name = cfg.get('NAME') or cfg.get('type')
    if not name:
        raise KeyError(
            "Model config must contain a 'NAME' field "
            "(e.g. NAME: Point_MAE_Mamba_serializationV2).")

    # Look up the class in the registry.
    # registry.Registry typically exposes the class dict as ._module_dict;
    # we try .get() first (cleaner API), then fall back to _module_dict.
    cls = None
    if hasattr(MODELS, 'get'):
        cls = MODELS.get(name)
    if cls is None and hasattr(MODELS, '_module_dict'):
        cls = MODELS._module_dict.get(name)

    if cls is not None and hasattr(cls, 'build_from_cfg'):
        return cls.build_from_cfg(cfg, **kwargs)

    # Fallback: standard registry build (expects cfg['type'] = class name)
    return MODELS.build(cfg, **kwargs)
