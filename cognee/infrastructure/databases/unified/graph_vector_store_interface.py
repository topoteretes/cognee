from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability


class GraphVectorStoreInterface:
    """
    Defines graph-native delete and rollback operations across graph and vector storage.

    Default implementations raise UnsupportedProvenanceCapability so existing
    adapters keep their current behavior until they explicitly implement this
    contract.
    """

    async def delete_by_source_ref(self, source_ref_key: str) -> None:
        """
        Delete artifacts owned only by the given source ref.

        Parameters:
        -----------

            - source_ref_key (str): Stable key for one dataset/data item pair.
        """
        raise UnsupportedProvenanceCapability()

    async def delete_by_dataset_id(self, dataset_id: str) -> None:
        """
        Delete artifacts owned only by the given dataset.

        Parameters:
        -----------

            - dataset_id (str): Unique identifier of the dataset being deleted.
        """
        raise UnsupportedProvenanceCapability()

    async def rollback_by_pipeline_run_id(self, pipeline_run_id: str) -> None:
        """
        Remove source refs attached by a failed pipeline run.

        Parameters:
        -----------

            - pipeline_run_id (str): Unique identifier of the pipeline run to roll back.
        """
        raise UnsupportedProvenanceCapability()
