from importlib import import_module


router_module = import_module("cognee.api.v1.users.routers.get_configuration_router")
StorePrincipalConfigurationPayloadDTO = router_module.StorePrincipalConfigurationPayloadDTO


def test_store_configuration_payload_fields_are_scalars():
    """The DTO fields must be plain scalars, not 1-tuples wrapping Form()."""
    payload = StorePrincipalConfigurationPayloadDTO(
        name="default_llm_settings",
        config={"temperature": 0.2},
    )

    assert payload.name == "default_llm_settings"
    assert isinstance(payload.name, str)
    assert payload.config == {"temperature": 0.2}
    assert isinstance(payload.config, dict)


def test_store_configuration_payload_field_annotations_are_scalars():
    """Field annotations must be str/dict, never a tuple type with a Form default."""
    fields = StorePrincipalConfigurationPayloadDTO.model_fields

    assert fields["name"].annotation is str
    assert fields["config"].annotation is dict
    # No default value should have been baked in as a tuple.
    assert not isinstance(fields["name"].default, tuple)
    assert not isinstance(fields["config"].default, tuple)
