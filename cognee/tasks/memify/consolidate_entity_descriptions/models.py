from pydantic import BaseModel


class NodeDescription(BaseModel):
    description: str


class MemberIsAText(BaseModel):
    member_name: str
    is_a_text: str


class EntityIsATexts(BaseModel):
    is_a_texts: list[MemberIsAText] = []
