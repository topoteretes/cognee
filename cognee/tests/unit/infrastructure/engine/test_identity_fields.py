from uuid import UUID, uuid4

from cognee.infrastructure.engine.models.DataPoint import DataPoint


class PersonWithIdentity(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class DepartmentWithIdentity(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class MultiFieldIdentity(DataPoint):
    first_name: str
    last_name: str
    metadata: dict = {
        "index_fields": ["first_name"],
        "identity_fields": ["first_name", "last_name"],
    }


class NoIdentityFields(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class PartialIdentity(DataPoint):
    name: str
    age: int = 0
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name", "age"]}


class TestSameValuesSameUUID:
    def test_same_name_produces_same_id(self):
        p1 = PersonWithIdentity(name="John")
        p2 = PersonWithIdentity(name="John")
        assert p1.id == p2.id

    def test_deterministic_across_calls(self):
        ids = [PersonWithIdentity(name="Alice").id for _ in range(10)]
        assert len(set(ids)) == 1


class TestDifferentValuesDifferentUUID:
    def test_different_names_produce_different_ids(self):
        p1 = PersonWithIdentity(name="John")
        p2 = PersonWithIdentity(name="Jane")
        assert p1.id != p2.id


class TestCrossTypeSafety:
    def test_same_name_different_class_different_id(self):
        person = PersonWithIdentity(name="Engineering")
        department = DepartmentWithIdentity(name="Engineering")
        assert person.id != department.id


class TestExplicitIdOverride:
    def test_explicit_id_takes_precedence(self):
        explicit_id = uuid4()
        p = PersonWithIdentity(id=explicit_id, name="John")
        assert p.id == explicit_id

    def test_explicit_id_differs_from_generated(self):
        explicit_id = uuid4()
        p_explicit = PersonWithIdentity(id=explicit_id, name="John")
        p_generated = PersonWithIdentity(name="John")
        assert p_explicit.id != p_generated.id


class TestMissingIdentityFieldFallback:
    def test_missing_field_falls_back_to_uuid4(self):
        """When identity_fields references a field not in the data, fall back to UUID4."""
        p1 = PartialIdentity(name="John")
        p2 = PartialIdentity(name="John")
        # age has a default so it IS in data; both should match
        assert p1.id == p2.id

    def test_truly_missing_field(self):
        """A class where identity_fields references a non-existent field."""

        class BadIdentity(DataPoint):
            name: str
            metadata: dict = {
                "index_fields": ["name"],
                "identity_fields": ["name", "nonexistent"],
            }

        b1 = BadIdentity(name="John")
        b2 = BadIdentity(name="John")
        # nonexistent is not in data, so falls back to UUID4 - different each time
        assert b1.id != b2.id


class TestNoIdentityFieldsBackwardCompat:
    def test_no_identity_fields_produces_random_uuid(self):
        n1 = NoIdentityFields(name="John")
        n2 = NoIdentityFields(name="John")
        assert n1.id != n2.id

    def test_base_datapoint_no_identity_fields(self):
        dp1 = DataPoint()
        dp2 = DataPoint()
        assert dp1.id != dp2.id


class TestMultiFieldIdentity:
    def test_same_multi_fields_same_id(self):
        m1 = MultiFieldIdentity(first_name="John", last_name="Doe")
        m2 = MultiFieldIdentity(first_name="John", last_name="Doe")
        assert m1.id == m2.id

    def test_different_multi_fields_different_id(self):
        m1 = MultiFieldIdentity(first_name="John", last_name="Doe")
        m2 = MultiFieldIdentity(first_name="John", last_name="Smith")
        assert m1.id != m2.id

    def test_field_order_matters(self):
        """first_name='A', last_name='B' should differ from first_name='B', last_name='A'."""
        m1 = MultiFieldIdentity(first_name="A", last_name="B")
        m2 = MultiFieldIdentity(first_name="B", last_name="A")
        assert m1.id != m2.id


class TestStringNormalization:
    def test_case_insensitive(self):
        p1 = PersonWithIdentity(name="John")
        p2 = PersonWithIdentity(name="JOHN")
        assert p1.id == p2.id

    def test_spaces_normalized(self):
        p1 = PersonWithIdentity(name="John Doe")
        p2 = PersonWithIdentity(name="John_Doe")
        assert p1.id == p2.id

    def test_apostrophes_removed(self):
        p1 = PersonWithIdentity(name="O'Brien")
        p2 = PersonWithIdentity(name="OBrien")
        assert p1.id == p2.id


class TestIdIsUUID5:
    def test_generated_id_is_valid_uuid(self):
        p = PersonWithIdentity(name="Test")
        assert isinstance(p.id, UUID)
        # UUID5 has version == 5
        assert p.id.version == 5

    def test_no_identity_id_is_uuid4(self):
        n = NoIdentityFields(name="Test")
        assert isinstance(n.id, UUID)
        assert n.id.version == 4


class TestTypeFieldPreserved:
    def test_type_is_class_name(self):
        p = PersonWithIdentity(name="John")
        assert p.type == "PersonWithIdentity"

    def test_type_not_affected_by_identity_fields(self):
        n = NoIdentityFields(name="John")
        assert n.type == "NoIdentityFields"
