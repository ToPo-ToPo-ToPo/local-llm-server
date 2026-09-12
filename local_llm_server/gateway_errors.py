"""Exceptions shared by gateway model management and HTTP routing."""


class CapacityError(RuntimeError):
    """No model slot or memory budget became available before the deadline."""


class GatewayDraining(RuntimeError):
    """The gateway is quiescing for a zero-drop restart."""
