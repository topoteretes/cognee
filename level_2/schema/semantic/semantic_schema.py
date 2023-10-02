from marshmallow import Schema, fields

class DocumentMetadataSchemaV1(Schema):
    user_id = fields.Str(required=True)
    memory_id = fields.Str(required=True)
    ltm_memory_id = fields.Str(required=True)
    st_memory_id = fields.Str(required=True)
    buffer_id = fields.Str(required=True)
    version = fields.Str(load_default="")
    agreement_id = fields.Str(load_default="")
    privacy_policy = fields.Str(load_default="")
    terms_of_service = fields.Str(load_default="")
    format = fields.Str(load_default="")
    schema_version = fields.Str(load_default="")
    checksum = fields.Str(load_default="")
    owner = fields.Str(load_default="")
    license = fields.Str(load_default="")
    validity_start = fields.Str(load_default="")
    validity_end = fields.Str(load_default="")

class DocumentMetadataSchemaV2(Schema):
    user_id = fields.Str(required=True)
    memory_id = fields.Str(required=True)
    ltm_memory_id = fields.Str(required=True)
    st_memory_id = fields.Str(required=True)
    buffer_id = fields.Str(required=True)
    version = fields.Str(load_default="")
    agreement_id = fields.Str(load_default="")
    privacy_policy = fields.Str(load_default="")
    terms_of_service = fields.Str(load_default="")
    format = fields.Str(load_default="")
    schema_version = fields.Str(load_default="")
    checksum = fields.Str(load_default="")
    owner = fields.Str(load_default="")
    license = fields.Str(load_default="")
    validity_start = fields.Str(load_default="")
    validity_end = fields.Str(load_default="")
    random = fields.Str(load_default="")

class DocumentSchema(Schema):
    metadata = fields.Nested(DocumentMetadataSchemaV1, required=True)
    page_content = fields.Str(required=True)


SCHEMA_VERSIONS = {
    "1.0": DocumentMetadataSchemaV1,
    "2.0": DocumentMetadataSchemaV2
}

def get_schema_version(version):
    return SCHEMA_VERSIONS.get(version, DocumentMetadataSchemaV1)