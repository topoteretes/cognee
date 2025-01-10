import uuid
from unittest.mock import patch

from cognee.modules.data.processing.document_types.ImageDocument import ImageDocument

GROUND_TRUTH = [
    {"word_count": 51, "len_text": 298, "cut_type": "sentence_end"},
    {"word_count": 62, "len_text": 369, "cut_type": "sentence_end"},
    {"word_count": 44, "len_text": 294, "cut_type": "sentence_end"},
]

TEST_TEXT = """A dramatic confrontation unfolds as a red fox and river otter engage in an energetic wrestling match at the water's edge. The fox, teeth bared in a playful snarl, has its front paws locked with the otter's flippers as they roll through the shallow stream, sending water spraying in all directions. The otter, displaying its surprising agility on land, counters by twisting its sleek body and attempting to wrap itself around the fox's shoulders, its whiskered face inches from the fox's muzzle.
The commotion has attracted an audience: a murder of crows has gathered in the low branches, their harsh calls adding to the chaos as they hop excitedly from limb to limb. One particularly bold crow dive-bombs the wrestling pair, causing both animals to momentarily freeze mid-tussle, creating a perfect snapshot of suspended action—the fox's fur dripping wet, the otter's body coiled like a spring, and the crow's wings spread wide against the golden morning light."""


def test_ImageDocument():
    document = ImageDocument(
        id=uuid.uuid4(),
        name="image-dummy-test",
        raw_data_location="",
        metadata_id=uuid.uuid4(),
        mime_type="",
    )
    with patch.object(ImageDocument, "transcribe_image", return_value=TEST_TEXT):
        for ground_truth, paragraph_data in zip(
            GROUND_TRUTH, document.read(chunk_size=64, chunker="text_chunker")
        ):
            assert ground_truth["word_count"] == paragraph_data.word_count, (
                f'{ground_truth["word_count"] = } != {paragraph_data.word_count = }'
            )
            assert ground_truth["len_text"] == len(paragraph_data.text), (
                f'{ground_truth["len_text"] = } != {len(paragraph_data.text) = }'
            )
            assert ground_truth["cut_type"] == paragraph_data.cut_type, (
                f'{ground_truth["cut_type"] = } != {paragraph_data.cut_type = }'
            )
