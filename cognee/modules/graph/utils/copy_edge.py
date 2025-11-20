def copy_edge(obj, overrides):
    cls = obj.__class__

    data = {c.name: getattr(obj, c.name) for c in cls.__table__.columns if not c.primary_key}
    data.update(overrides)

    return cls(**data)
