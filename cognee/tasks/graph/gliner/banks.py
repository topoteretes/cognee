"""Frozen starter label banks used by ``schema_from_label_bank``.

The names below are the *only* labels bank probing may return. They are fixed
in the repository on purpose: when neither the caller nor an ontology supplies
a schema, the type vocabulary of a GLiNER-built graph is chosen from this list
by measuring which labels actually fire on the first batch — never invented at
runtime. Descriptions are passed to GLiNER as label descriptions.
"""

from types import MappingProxyType

LABEL_BANK = MappingProxyType(
    {
        "person": "Full name of a human being",
        "organization": "Company, institution, agency, team, or other named group",
        "location": "City, country, region, address, or other named place",
        "event": "Named happening such as a conference, election, war, or launch",
        "date": "Calendar date or specific point in time",
        "time_period": "Span of time such as a year, quarter, decade, or era",
        "product": "Commercial product or service",
        "technology": "Technology, system, platform, or technical standard",
        "software": "Software application, library, or framework",
        "programming_language": "Programming or markup language",
        "concept": "Abstract idea, method, or field of study",
        "document": "Report, article, paper, book, contract, or other named document",
        "project": "Named project, initiative, or programme",
        "role": "Job title or position held by a person",
        "industry": "Economic sector or industry",
        "money": "Monetary amount with or without a currency",
        "quantity": "Numeric measurement or amount with a unit",
        "percentage": "Percentage value",
        "law": "Law, regulation, treaty, or legal case",
        "disease": "Disease, medical condition, or symptom",
        "drug": "Medication, drug, or compound used as a treatment",
        "chemical": "Chemical element, compound, or material",
        "animal": "Animal species or named animal",
        "plant": "Plant species",
        "artwork": "Book, film, song, painting, or other creative work",
        "award": "Prize, award, or honour",
        "facility": "Building, factory, airport, hospital, or other physical facility",
        "vehicle": "Car, aircraft, ship, spacecraft, or other vehicle",
        "food": "Food or drink",
        "nationality": "Nationality, ethnic, religious, or political group",
    }
)

RELATION_BANK = MappingProxyType(
    {
        "works_for": "Person is employed by or works at an organization",
        "founded_by": "Organization was founded by a person or organization",
        "located_in": "Entity is physically located in a place",
        "headquartered_in": "Organization has its headquarters in a place",
        "part_of": "Entity is a component or subdivision of another entity",
        "member_of": "Person or organization is a member of a group",
        "owns": "Entity owns or controls another entity",
        "acquired": "Organization acquired another organization",
        "produces": "Organization makes a product or technology",
        "uses": "Entity uses a product, technology, or method",
        "created_by": "Work, product, or concept was created by a person or organization",
        "leads": "Person leads or manages an organization, team, or project",
        "collaborates_with": "Two entities work together",
        "born_in": "Person was born in a place",
        "participates_in": "Entity takes part in an event or project",
        "occurred_on": "Event happened on a date or during a time period",
    }
)
