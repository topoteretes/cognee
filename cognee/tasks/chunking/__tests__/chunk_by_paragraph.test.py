from cognee.tasks.chunking import chunk_by_paragraph

if __name__ == "__main__":
    def test_chunking_on_whole_text():
        test_text = """This is example text. It contains multiple sentences.
        This is a second paragraph. First two paragraphs are whole.
        Third paragraph is a bit longer and is finished with a dot."""

        chunks = []

        for chunk_data in chunk_by_paragraph(test_text, 12, batch_paragraphs = False):
            chunks.append(chunk_data)

        assert len(chunks) == 3

        assert chunks[0]["text"] == "This is example text. It contains multiple sentences."
        assert chunks[0]["word_count"] == 8
        assert chunks[0]["cut_type"] == "paragraph_end"

        assert chunks[1]["text"] == "This is a second paragraph. First two paragraphs are whole."
        assert chunks[1]["word_count"] == 10
        assert chunks[1]["cut_type"] == "paragraph_end"

        assert chunks[2]["text"] == "Third paragraph is a bit longer and is finished with a dot."
        assert chunks[2]["word_count"] == 12
        assert chunks[2]["cut_type"] == "sentence_end"

    def test_chunking_on_cut_text():
        test_text = """This is example text. It contains multiple sentences.
        This is a second paragraph. First two paragraphs are whole.
        Third paragraph is cut and is missing the dot at the end"""

        chunks = []

        for chunk_data in chunk_by_paragraph(test_text, 12, batch_paragraphs = False):
            chunks.append(chunk_data)

        assert len(chunks) == 3

        assert chunks[0]["text"] == "This is example text. It contains multiple sentences."
        assert chunks[0]["word_count"] == 8
        assert chunks[0]["cut_type"] == "paragraph_end"

        assert chunks[1]["text"] == "This is a second paragraph. First two paragraphs are whole."
        assert chunks[1]["word_count"] == 10
        assert chunks[1]["cut_type"] == "paragraph_end"

        assert chunks[2]["text"] == "Third paragraph is cut and is missing the dot at the end"
        assert chunks[2]["word_count"] == 12
        assert chunks[2]["cut_type"] == "sentence_cut"

    test_chunking_on_whole_text()
    test_chunking_on_cut_text()
