"""Code graph extraction tasks backed by enola.

These tasks build an architectural knowledge graph of a code repository using
enola (https://github.com/enola-labs/enola) by Enola Labs, licensed under
Apache-2.0. Cognee invokes the external enola binary (`enola --generate`) and
parses its documented snapshot contract (`.enola/facts.jsonl`,
`insights.json`, `receipt.json` — docs/schema/ upstream, format_version 1);
no enola code is vendored into cognee.
"""

from .enola import (
    SUPPORTED_FORMAT_VERSIONS,
    EnolaNotInstalledError,
    EnolaSnapshotError,
    find_enola_binary,
    is_enola_id,
    normalize_relation,
    parse_enola_snapshot,
    relation_target_id,
    run_enola_generate,
    validate_receipt,
)
from .install_enola import (
    ENOLA_PINNED_VERSION,
    EnolaInstallError,
    install_enola,
)
from .extract_code_graph import (
    add_code_graph_edges,
    build_code_graph_edges,
    extract_code_graph,
    get_code_graph_tasks,
    map_facts_to_data_points,
    receipt_projection,
)
from .models import (
    ApiEndpoint,
    CodeAssociation,
    CodeExtractionAccount,
    CodeGraphEntity,
    CodeInsight,
    CodeIntent,
    CodeLintFinding,
    CodeModule,
    CodeRepository,
    CodeService,
    CodeSymbol,
    CodeTestReference,
    CodeFileReference,
    ExternalDependency,
    StorageResource,
)
