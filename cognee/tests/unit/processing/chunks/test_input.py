import pytest

INPUT_TEXTS = {
    "empty": "",
    "single_char": "x",
    "whitespace": "   \n\t   \r\n   ",
    "unicode_special": "Hello 👋 مرحبا שָׁלוֹם",
    "mixed_endings": "line1\r\nline2\nline3\r\nline4",
    "many_newlines": "\n\n\n\ntext\n\n\n\n",
    "html_mixed": "<p>Hello</p>\nPlain text\n<div>World</div>",
    "urls_emails": "Visit https://example.com or email user@example.com",
    "elipses": "Hello...How are you…",
    "english_lists": """Let me think through the key attributes that would be important to test in a text chunking system.
Here are the essential attributes to test:

Chunking Boundaries Accuracy:


Proper sentence boundary detection
Handling of punctuation marks
Recognition of paragraph breaks
Treatment of special characters and whitespace
Proper handling of quotes and nested text structures


Language Support:


Handling of different languages and scripts
Support for multilingual documents
Proper Unicode handling
Treatment of language-specific punctuation


Special Cases Handling:


Lists and bullet points
Tables and structured content
Code blocks or technical content
Citations and references
Headers and footers
URLs and email addresses


Performance Metrics:


Processing speed for different text lengths
Memory usage with large documents
Scalability with increasing document size
Consistency across multiple runs


Document Format Support:


Plain text handling
HTML/XML content
PDF text extraction
Markdown formatting
Mixed format documents


Error Handling:


Malformed input text
Incomplete sentences
Truncated documents
Invalid characters
Missing punctuation


Configuration Flexibility:


Adjustable chunk sizes
Customizable boundary rules
Configurable overlap between chunks
Token vs. character-based chunking options


Preservation of Context:


Maintaining semantic coherence
Preserving contextual relationships
Handling cross-references
Maintaining document structure

Would you like me to elaborate on any of these attributes or discuss specific testing strategies for them?""",
    "python_code": """from typing import (
    Literal as L,
    Any,
    TypeAlias,
    overload,
    TypeVar,
    Protocol,
    type_check_only,
)

from numpy import generic

from numpy._typing import (
    ArrayLike,
    NDArray,
    _ArrayLikeInt,
    _ArrayLike,
)

__all__ = ["pad"]

_SCT = TypeVar("_SCT", bound=generic)

@type_check_only
class _ModeFunc(Protocol):
    def __call__(
        self,
        vector: NDArray[Any],
        iaxis_pad_width: tuple[int, int],
        iaxis: int,
        kwargs: dict[str, Any],
        /,
    ) -> None: ...

_ModeKind: TypeAlias = L[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
    "empty",
]


# TODO: In practice each keyword argument is exclusive to one or more
# specific modes. Consider adding more overloads to express this in the future.

# Expand `**kwargs` into explicit keyword-only arguments
@overload
def pad(
    array: _ArrayLike[_SCT],
    pad_width: _ArrayLikeInt,
    mode: _ModeKind = ...,
    *,
    stat_length: None | _ArrayLikeInt = ...,
    constant_values: ArrayLike = ...,
    end_values: ArrayLike = ...,
    reflect_type: L["odd", "even"] = ...,
) -> NDArray[_SCT]: ...
@overload
def pad(
    array: ArrayLike,
    pad_width: _ArrayLikeInt,
    mode: _ModeKind = ...,
    *,
    stat_length: None | _ArrayLikeInt = ...,
    constant_values: ArrayLike = ...,
    end_values: ArrayLike = ...,
    reflect_type: L["odd", "even"] = ...,
) -> NDArray[Any]: ...
@overload
def pad(
    array: _ArrayLike[_SCT],
    pad_width: _ArrayLikeInt,
    mode: _ModeFunc,
    **kwargs: Any,
) -> NDArray[_SCT]: ...
@overload
def pad(
    array: ArrayLike,
    pad_width: _ArrayLikeInt,
    mode: _ModeFunc,
    **kwargs: Any,
) -> NDArray[Any]: ...""",
    "english_text": """O for that warning voice, which he who saw
Th' Apocalyps, heard cry in Heaven aloud,
Then when the Dragon, put to second rout,
Came furious down to be reveng'd on men,
Wo to the inhabitants on Earth! that now, [ 5 ]
While time was, our first-Parents had bin warnd
The coming of thir secret foe, and scap'd
Haply so scap'd his mortal snare; for now
Satan, now first inflam'd with rage, came down,
The Tempter ere th' Accuser of man-kind, [ 10 ]
To wreck on innocent frail man his loss
Of that first Battel, and his flight to Hell:
Yet not rejoycing in his speed, though bold,
Far off and fearless, nor with cause to boast,
Begins his dire attempt, which nigh the birth [ 15 ]
Now rowling, boiles in his tumultuous brest,
And like a devillish Engine back recoiles
Upon himself; horror and doubt distract
His troubl'd thoughts, and from the bottom stirr
The Hell within him, for within him Hell [ 20 ]
He brings, and round about him, nor from Hell
One step no more then from himself can fly
By change of place: Now conscience wakes despair
That slumberd, wakes the bitter memorie
Of what he was, what is, and what must be [ 25 ]
Worse; of worse deeds worse sufferings must ensue.
Sometimes towards Eden which now in his view
Lay pleasant, his grievd look he fixes sad,
Sometimes towards Heav'n and the full-blazing Sun,
Which now sat high in his Meridian Towre: [ 30 ]
Then much revolving, thus in sighs began.

O thou that with surpassing Glory crownd,
Look'st from thy sole Dominion like the God
Of this new World; at whose sight all the Starrs
Hide thir diminisht heads; to thee I call, [ 35 ]
But with no friendly voice, and add thy name
O Sun, to tell thee how I hate thy beams
That bring to my remembrance from what state
I fell, how glorious once above thy Spheare;
Till Pride and worse Ambition threw me down [ 40 ]
Warring in Heav'n against Heav'ns matchless King:
Ah wherefore! he deservd no such return
From me, whom he created what I was
In that bright eminence, and with his good
Upbraided none; nor was his service hard. [ 45 ]
What could be less then to afford him praise,
The easiest recompence, and pay him thanks,
How due! yet all his good prov'd ill in me,
And wrought but malice; lifted up so high
I sdeind subjection, and thought one step higher [ 50 ]
Would set me highest, and in a moment quit
The debt immense of endless gratitude,
So burthensome, still paying, still to ow;
Forgetful what from him I still receivd,
And understood not that a grateful mind [ 55 ]
By owing owes not, but still pays, at once
Indebted and dischargd; what burden then?
O had his powerful Destiny ordaind
Me some inferiour Angel, I had stood
Then happie; no unbounded hope had rais'd [ 60 ]
Ambition. Yet why not? som other Power
As great might have aspir'd, and me though mean
Drawn to his part; but other Powers as great
Fell not, but stand unshak'n, from within
Or from without, to all temptations arm'd. [ 65 ]
Hadst thou the same free Will and Power to stand?
Thou hadst: whom hast thou then or what to accuse,
But Heav'ns free Love dealt equally to all?
Be then his Love accurst, since love or hate,
To me alike, it deals eternal woe. [ 70 ]
Nay curs'd be thou; since against his thy will
Chose freely what it now so justly rues.
Me miserable! which way shall I flie
Infinite wrauth, and infinite despaire?
Which way I flie is Hell; my self am Hell; [ 75 ]
And in the lowest deep a lower deep
Still threatning to devour me opens wide,
To which the Hell I suffer seems a Heav'n.
O then at last relent: is there no place
Left for Repentance, none for Pardon left? [ 80 ]
None left but by submission; and that word
Disdain forbids me, and my dread of shame
Among the Spirits beneath, whom I seduc'd
With other promises and other vaunts
Then to submit, boasting I could subdue [ 85 ]
Th' Omnipotent. Ay me, they little know
How dearly I abide that boast so vaine,
Under what torments inwardly I groane:
While they adore me on the Throne of Hell,
With Diadem and Sceptre high advanc'd [ 90 ]
The lower still I fall, onely Supream
In miserie; such joy Ambition findes.
But say I could repent and could obtaine
By Act of Grace my former state; how soon
Would higth recall high thoughts, how soon unsay [ 95 ]
What feign'd submission swore: ease would recant
Vows made in pain, as violent and void.
For never can true reconcilement grow
Where wounds of deadly hate have peirc'd so deep:
Which would but lead me to a worse relapse [ 100 ]""",
}

INPUT_TEXTS_LONGWORDS = {
    "chinese_text": """在这个繁华的城市里，藏着一个古老的小巷，名叫杨柳巷。巷子两旁的青石板路已经被无数行人的脚步磨得发亮，斑驳的老墙上爬满了常青藤，给这个充满历史气息的小巷增添了一抹生机。每天清晨，巷子里都会飘出阵阵香气，那是张婆婆家的早点铺子散发出的包子和豆浆的味道。老店门前经常排着长队，有步履匆匆的上班族，也有悠闲散步的老人。巷子深处有一家传统的茶馆，古色古香的木桌椅上总是坐满了品茶聊天的街坊邻里。傍晚时分，夕阳的余晖洒在石板路上，为这个充满生活气息的小巷染上一层温暖的金色。街角的老榕树下，常常有卖唱的艺人在这里驻足，用沧桑的嗓音讲述着这座城市的故事。偶尔，还能看到三三两两的游客举着相机，试图捕捉这里独特的市井风情。这条看似普通的小巷，承载着太多市民的回忆和岁月的痕迹，它就像是这座城市的一个缩影，悄悄地诉说着曾经的故事。""",
}
