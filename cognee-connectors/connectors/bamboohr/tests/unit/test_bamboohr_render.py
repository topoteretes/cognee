"""The rendering: one frozen, deterministic text per employee, allowlisted fields only."""

from cognee_connector_bamboohr.source import render_employee

FIELDS = ["department", "firstName", "jobTitle", "lastName", "status"]
LABELS = {
    "firstName": "First Name",
    "lastName": "Last Name",
    "jobTitle": "Job Title",
    "department": "Department",
    "status": "Status",
}


def test_a_named_employee_renders_an_intro_and_one_line_per_set_field():
    row = render_employee(
        "7",
        {
            "firstName": "Ada",
            "lastName": "Lovelace",
            "jobTitle": "Engineer",
            "department": "Research",
            "status": "Active",
            "location": "London",  # not in the allowlist: never rendered
        },
        fields=FIELDS,
        labels=LABELS,
        subdomain="acme",
    )
    assert row == {
        "id": "employee:7",
        "title": "Ada Lovelace",
        "content": (
            "BambooHR employee Ada Lovelace.\n"
            "Department: Research\n"
            "First Name: Ada\n"
            "Job Title: Engineer\n"
            "Last Name: Lovelace\n"
            "Status: Active"
        ),
        "url": "https://acme.bamboohr.com/employees/view/7",
        "_deleted": False,
    }


def test_lines_follow_the_sorted_allowlist_order_not_the_records_order():
    row = render_employee(
        "7",
        {"status": "Active", "firstName": "Ada", "lastName": "Lovelace"},
        fields=FIELDS,
        labels=LABELS,
        subdomain="acme",
    )
    assert row["content"].splitlines() == [
        "BambooHR employee Ada Lovelace.",
        "First Name: Ada",
        "Last Name: Lovelace",
        "Status: Active",
    ]


def test_an_empty_allowlisted_field_renders_no_line():
    row = render_employee(
        "7",
        {"firstName": "Ada", "lastName": "Lovelace", "department": None, "jobTitle": "  "},
        fields=FIELDS,
        labels=LABELS,
        subdomain="acme",
    )
    assert "Department:" not in row["content"]
    assert "Job Title:" not in row["content"]


def test_a_multi_value_field_renders_sorted_so_the_text_is_deterministic():
    row = render_employee(
        "7",
        {"firstName": "Ada", "lastName": "Lovelace", "department": ["Beta", "Alpha"]},
        fields=FIELDS,
        labels=LABELS,
        subdomain="acme",
    )
    assert "Department: Alpha, Beta" in row["content"]


def test_an_employee_without_a_name_is_titled_by_its_id():
    row = render_employee(
        "7", {"jobTitle": "Engineer"}, fields=FIELDS, labels=LABELS, subdomain="acme"
    )
    assert row["title"] == "Employee 7"
    assert row["content"].startswith("BambooHR employee 7 has no name in the synced fields.")


def test_a_field_bamboohr_lists_without_a_display_name_is_labeled_by_its_alias():
    row = render_employee(
        "7",
        {"firstName": "Ada", "lastName": "Lovelace", "customTShirtSize": "M"},
        fields=["customTShirtSize", "firstName", "lastName"],
        labels={
            "customTShirtSize": "customTShirtSize",
            "firstName": "First Name",
            "lastName": "Last Name",
        },
        subdomain="acme",
    )
    assert "customTShirtSize: M" in row["content"]
