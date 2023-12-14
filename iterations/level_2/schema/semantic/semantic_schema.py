from marshmallow import Schema, fields

class DocumentMetadataSchemaV1(Schema):
    user_id = fields.Str(required=True)
    memory_id = fields.Str(required=True)
    ltm_memory_id = fields.Str(required=True)
    st_memory_id = fields.Str(required=True)
    buffer_id = fields.Str(required=True)
    version = fields.Str(missing="")
    agreement_id = fields.Str(missing="")
    privacy_policy = fields.Str(missing="")
    terms_of_service = fields.Str(missing="")
    format = fields.Str(missing="")
    schema_version = fields.Str(missing="")
    checksum = fields.Str(missing="")
    owner = fields.Str(missing="")
    license = fields.Str(missing="")
    validity_start = fields.Str(missing="")
    validity_end = fields.Str(missing="")

class DocumentMetadataSchemaV2(Schema):
    user_id = fields.Str(required=True)
    memory_id = fields.Str(required=True)
    ltm_memory_id = fields.Str(required=True)
    st_memory_id = fields.Str(required=True)
    buffer_id = fields.Str(required=True)
    version = fields.Str(missing="")
    agreement_id = fields.Str(missing="")
    privacy_policy = fields.Str(missing="")
    terms_of_service = fields.Str(missing="")
    format = fields.Str(missing="")
    schema_version = fields.Str(missing="")
    checksum = fields.Str(missing="")
    owner = fields.Str(missing="")
    license = fields.Str(missing="")
    validity_start = fields.Str(missing="")
    validity_end = fields.Str(missing="")
    random = fields.Str(missing="")

class DocumentSchema(Schema):
    metadata = fields.Nested(DocumentMetadataSchemaV1, required=True)
    page_content = fields.Str(required=True)


SCHEMA_VERSIONS = {
    "1.0": DocumentMetadataSchemaV1,
    "2.0": DocumentMetadataSchemaV2
}

def get_schema_version(version):
    return SCHEMA_VERSIONS.get(version, DocumentMetadataSchemaV1)