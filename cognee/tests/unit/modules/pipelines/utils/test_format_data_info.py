import unittest
from uuid import uuid4
from cognee.modules.data.models import Data
from cognee.modules.pipelines.utils.format_data_info import format_data_info


class TestFormatDataInfo(unittest.TestCase):
    def test_format_data_info_none(self):
        self.assertEqual(format_data_info(None), "None")
        self.assertEqual(format_data_info(""), "None")
        self.assertEqual(format_data_info([]), "None")

    def test_format_data_info_data_list(self):
        data_list = [Data(id=uuid4()), Data(id=uuid4())]
        expected = [str(item.id) for item in data_list]
        self.assertEqual(format_data_info(data_list), expected)

    def test_format_data_info_short_string(self):
        self.assertEqual(format_data_info("short text"), "short text")

    def test_format_data_info_long_string(self):
        long_text = "a" * 1000
        result = format_data_info(long_text)
        self.assertTrue(result.startswith("a" * 500))
        self.assertTrue(result.endswith("... [truncated 1000 chars]"))
        self.assertEqual(len(result), 500 + len("... [truncated 1000 chars]"))
