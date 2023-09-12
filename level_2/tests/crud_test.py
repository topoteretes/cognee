import unittest
import asyncio

import sys
sys.path.append("..")  # Adds higher directory to python modules path.

from level_2.level_2_pdf_vectorstore__dlt_contracts import Memory
class TestMemory(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.get_event_loop()
        self.memory = Memory(user_id="123")
        self.loop.run_until_complete(self.memory.async_init())

    def test_add_fetch_delete_semantic_memory(self):
        async def semantic_workflow():
            params = {"sample_param": "value"}
            sample_memory = "sample semantic memory"

            # Add
            await self.memory._add_semantic_memory(sample_memory, params=params)
            # Fetch
            fetched = await self.memory._fetch_semantic_memory(sample_memory, params)
            fetched_text = fetched['data']['Get']['EPISODICMEMORY'][0]['text']
            self.assertIn(sample_memory, fetched_text)  # Replace this with the appropriate validation
            # Delete
            await self.memory._delete_semantic_memory()
            # Verify Deletion
            after_delete = await self.memory._fetch_semantic_memory(sample_memory, params)
            self.assertNotIn(sample_memory, after_delete)  # Replace with the appropriate validation

        self.loop.run_until_complete(semantic_workflow())

    def test_add_fetch_delete_episodic_memory(self):
        async def episodic_workflow():
            params = {"sample_param": "value"}
            sample_memory = """{
                                "sample_key": "sample_value"
                              }"""

            # Add
            await self.memory._add_episodic_memory(observation=sample_memory, params=params)
            # Fetch
            fetched = await self.memory._fetch_episodic_memory(sample_memory)
            fetched_text = fetched['data']['Get']['EPISODICMEMORY'][0]['text']
            self.assertIn(sample_memory, fetched_text)  # Replace this with the appropriate validation
            # Delete
            await self.memory._delete_episodic_memory()
            # Verify Deletion
            after_delete = await self.memory._fetch_episodic_memory(sample_memory)
            self.assertNotIn(sample_memory, after_delete)  # Replace with the appropriate validation

        self.loop.run_until_complete(episodic_workflow())

    # def test_add_fetch_delete_buffer_memory(self):
    #     async def buffer_workflow():
    #         params = {"sample_param": "value"}
    #         user_input = "sample buffer input"
    #         namespace = "sample_namespace"
    #
    #         # Add
    #         await self.memory._add_buffer_memory(user_input=user_input, namespace=namespace, params=params)
    #         # Fetch
    #         fetched = await self.memory._fetch_buffer_memory(user_input, namespace)
    #         self.assertIn(user_input, fetched)  # Replace this with the appropriate validation
    #         # Delete
    #         await self.memory._delete_buffer_memory()
    #         # Verify Deletion
    #         after_delete = await self.memory._fetch_buffer_memory(user_input, namespace)
    #         self.assertNotIn(user_input, after_delete)  # Replace with the appropriate validation
    #
    #     self.loop.run_until_complete(buffer_workflow())


if __name__ == '__main__':
    unittest.main()


