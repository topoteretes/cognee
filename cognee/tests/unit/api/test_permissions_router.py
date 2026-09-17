from cognee.api.v1.permissions.routers.get_permissions_router import get_permissions_router


def test_principal_datasets_route_is_registered():
    routes = {route.name: route for route in get_permissions_router().routes}

    route = routes["get_principal_datasets"]

    assert route.path == "/principals/{principal_id}/datasets"
    assert "GET" in route.methods
