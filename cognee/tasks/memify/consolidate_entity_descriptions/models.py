from typing import List

from pydantic import BaseModel


class NodeDescription(BaseModel):
    description: str


class MemberIsAText(BaseModel):
    member_name: str
    is_a_text: str


class EntityTypeDescription(BaseModel):
    description: str
    is_a_texts: List[MemberIsAText] = []


class EntityIsATexts(BaseModel):
    is_a_texts: List[MemberIsAText] = []
