from cognee.tasks.chunks import chunk_by_paragraph

GROUND_TRUTH = {
    "whole_text": [
        {"text": "This is example text. It contains multiple sentences.", "word_count": 8, "cut_type": "paragraph_end"},
        {"text": "This is a second paragraph. First two paragraphs are whole.", "word_count": 10 , "cut_type": "paragraph_end"},
        {"text": "Third paragraph is a bit longer and is finished with a dot.", "word_count": 12, "cut_type": "sentence_end"}
    ],
    "cut_text": [
        {"text": "This is example text. It contains multiple sentences.", "word_count": 8, "cut_type": "paragraph_end"},
        {"text": "This is a second paragraph. First two paragraphs are whole.", "word_count": 10, "cut_type": "paragraph_end"},
        {"text": "Third paragraph is cut and is missing the dot at the end", "word_count": 12, "cut_type": "sentence_cut"}
    ]
}

INPUT_TEXT = {
    "whole_text": """This is example text. It contains multiple sentences.
    This is a second paragraph. First two paragraphs are whole.
    Third paragraph is a bit longer and is finished with a dot.""",
    "cut_text": """This is example text. It contains multiple sentences.
    This is a second paragraph. First two paragraphs are whole.
    Third paragraph is cut and is missing the dot at the end"""
}

def test_chunking(test_text, ground_truth):
    chunks = []
    for chunk_data in chunk_by_paragraph(test_text, 12, batch_paragraphs = False):
        chunks.append(chunk_data)

    assert len(chunks) == 3

    for ground_truth_item, chunk in zip(ground_truth, chunks):
        for key in ["text", "word_count", "cut_type"]:
            assert chunk[key] == ground_truth_item[key], f'{key = }: {chunk[key] = } != {ground_truth_item[key] = }'



if __name__ == "__main__":
    test_chunking(INPUT_TEXT["whole_text"], GROUND_TRUTH["whole_text"])
    test_chunking(INPUT_TEXT["cut_text"], GROUND_TRUTH["cut_text"])
