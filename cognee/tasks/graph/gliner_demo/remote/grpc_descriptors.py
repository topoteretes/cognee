"""Serialized ``FileDescriptorProto``s for the worker's gRPC contract and grpc.health.v1.

Compiled from ``proto/gliner_worker.proto`` and ``proto/health.proto``. The gRPC
transport builds its message classes from these at runtime instead of shipping
protoc-generated modules, because generated modules refuse to import on a
protobuf or grpcio runtime older than the protoc that produced them, and cognee
supports protobuf 5.29 upward. A unit test recompiles the files and compares.

After editing a .proto, regenerate this module (needs ``grpcio-tools``)::

    python -m cognee.tasks.graph.gliner_demo.remote.grpc_descriptors && ruff format <this file>
"""

from __future__ import annotations

from pathlib import Path

PROTO_DIR = Path(__file__).parent / "proto"
PROTO_FILES = {"GLINER_WORKER_PROTO": "gliner_worker.proto", "HEALTH_PROTO": "health.proto"}

# BEGIN GENERATED
GLINER_WORKER_PROTO = (
    b'\n\x13gliner_worker.proto\x12\x10gliner_worker.v1"\xa8\x01\n\x0eExtractRequest\x12/'
    b"\n\x06inputs\x18\x01 \x03(\x0b2\x17.gliner_worker.v1.InputR\x06inputs\x120\n\x06schema\x18\x02 "
    b"\x01(\x0b2\x18.gliner_worker.v1.SchemaR\x06schema\x123\n\x07options\x18\x03 \x01(\x0b2\x19.gli"
    b'ner_worker.v1.OptionsR\x07options"+\n\x05Input\x12\x0e\n\x02id\x18\x01 \x01(\tR\x02id\x12\x12\n\x04t'
    b'ext\x18\x02 \x01(\tR\x04text"t\n\x06Schema\x123\n\x08entities\x18\x01 \x03(\x0b2\x17.gliner_worker.'
    b"v1.LabelR\x08entities\x125\n\trelations\x18\x02 \x03(\x0b2\x17.gliner_worker.v1.Lab"
    b'elR\trelations"R\n\x05Label\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12%\n\x0bdescription\x18\x02 '
    b'\x01(\tH\x00R\x0bdescription\x88\x01\x01B\x0e\n\x0c_description"\xbc\x03\n\x07Options\x12!\n\tthresho'
    b'ld\x18\x01 \x01(\x01H\x00R\tthreshold\x88\x01\x01\x12"\n\nbatch_size\x18\x02 \x01(\x05H\x01R\tbatchSize\x88\x01\x01'
    b"\x122\n\x12include_confidence\x18\x03 \x01(\x08H\x02R\x11includeConfidence\x88\x01\x01\x12(\n\rincl"
    b"ude_spans\x18\x04 \x01(\x08H\x03R\x0cincludeSpans\x88\x01\x01\x12*\n\x0eoverlap_policy\x18\x05 \x01(\tH\x04"
    b"R\roverlapPolicy\x88\x01\x01\x12&\n\x0cwindow_words\x18\x06 \x01(\x05H\x05R\x0bwindowWords\x88\x01\x01\x125"
    b"\n\x14window_overlap_words\x18\x07 \x01(\x05H\x06R\x12windowOverlapWords\x88\x01\x01B\x0c\n\n_th"
    b"resholdB\r\n\x0b_batch_sizeB\x15\n\x13_include_confidenceB\x10\n\x0e_include_sp"
    b"ansB\x11\n\x0f_overlap_policyB\x0f\n\r_window_wordsB\x17\n\x15_window_overlap_w"
    b'ords"\x96\x01\n\x0fExtractResponse\x122\n\x05items\x18\x01 \x03(\x0b2\x1c.gliner_worker.v1.I'
    b"temResultR\x05items\x12\x14\n\x05model\x18\x02 \x01(\tR\x05model\x12\x1d\n\nelapsed_ms\x18\x03 \x01(\x01R\t"
    b'elapsedMs\x12\x1a\n\x08windowed\x18\x04 \x01(\x08R\x08windowed"\xe9\x02\n\nItemResult\x12\x0e\n\x02id\x18\x01'
    b" \x01(\tR\x02id\x12F\n\x08entities\x18\x02 \x03(\x0b2*.gliner_worker.v1.ItemResult.Ent"
    b"itiesEntryR\x08entities\x12I\n\trelations\x18\x03 \x03(\x0b2+.gliner_worker.v1.I"
    b"temResult.RelationsEntryR\trelations\x1aZ\n\rEntitiesEntry\x12\x10\n\x03key\x18"
    b"\x01 \x01(\tR\x03key\x123\n\x05value\x18\x02 \x01(\x0b2\x1d.gliner_worker.v1.MentionListR\x05va"
    b"lue:\x028\x01\x1a\\\n\x0eRelationsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x124\n\x05value\x18\x02 \x01(\x0b2\x1e"
    b'.gliner_worker.v1.RelationListR\x05value:\x028\x01"D\n\x0bMentionList\x125\n\x08'
    b'mentions\x18\x01 \x03(\x0b2\x19.gliner_worker.v1.MentionR\x08mentions"H\n\x0cRelat'
    b"ionList\x128\n\trelations\x18\x01 \x03(\x0b2\x1a.gliner_worker.v1.RelationR\trela"
    b'tions"\x95\x01\n\x07Mention\x12\x12\n\x04text\x18\x01 \x01(\tR\x04text\x12#\n\nconfidence\x18\x02 \x01(\x01H\x00R'
    b"\nconfidence\x88\x01\x01\x12\x19\n\x05start\x18\x03 \x01(\x05H\x01R\x05start\x88\x01\x01\x12\x15\n\x03end\x18\x04 \x01(\x05H\x02R\x03en"
    b'd\x88\x01\x01B\r\n\x0b_confidenceB\x08\n\x06_startB\x06\n\x04_end"h\n\x08Relation\x12-\n\x04head\x18\x01 '
    b"\x01(\x0b2\x19.gliner_worker.v1.MentionR\x04head\x12-\n\x04tail\x18\x02 \x01(\x0b2\x19.gliner_"
    b"worker.v1.MentionR\x04tail2^\n\x0cGlinerWorker\x12N\n\x07Extract\x12 .gliner_"
    b"worker.v1.ExtractRequest\x1a!.gliner_worker.v1.ExtractResponseb"
    b"\x06proto3"
)
HEALTH_PROTO = (
    b'\n\x0chealth.proto\x12\x0egrpc.health.v1".\n\x12HealthCheckRequest\x12\x18\n\x07serv'
    b'ice\x18\x01 \x01(\tR\x07service"\xb1\x01\n\x13HealthCheckResponse\x12I\n\x06status\x18\x01 \x01(\x0e21'
    b'.grpc.health.v1.HealthCheckResponse.ServingStatusR\x06status"O\n'
    b"\rServingStatus\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0b\n\x07SERVING\x10\x01\x12\x0f\n\x0bNOT_SERVING\x10\x02\x12\x13\n"
    b'\x0fSERVICE_UNKNOWN\x10\x032Z\n\x06Health\x12P\n\x05Check\x12".grpc.health.v1.Healt'
    b"hCheckRequest\x1a#.grpc.health.v1.HealthCheckResponseb\x06proto3"
)
# END GENERATED


def compile_proto(file_name: str) -> bytes:
    """Compile one file in ``proto/`` with grpcio-tools and serialize its descriptor."""
    import tempfile

    from google.protobuf import descriptor_pb2
    from grpc_tools import protoc

    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "set.pb"
        status = protoc.main(
            [
                "protoc",
                f"-I{PROTO_DIR}",
                f"--descriptor_set_out={out}",
                str(PROTO_DIR / file_name),
            ]
        )
        if status != 0:
            raise RuntimeError(f"protoc failed on {file_name} with status {status}")
        descriptor_set = descriptor_pb2.FileDescriptorSet.FromString(  # ty: ignore[unresolved-attribute] - generated, absent from the stubs
            out.read_bytes()
        )
    [file] = descriptor_set.file
    return file.SerializeToString()


def _render(name: str, data: bytes) -> str:
    # Implicit concatenation keeps every line within the 100-character limit.
    pieces = [repr(data[start : start + 60]) for start in range(0, len(data), 60)]
    return f"{name} = (\n" + "".join(f"    {piece}\n" for piece in pieces) + ")\n"


def regenerate() -> None:
    path = Path(__file__)
    source = path.read_text()
    start = source.index("# BEGIN GENERATED\n") + len("# BEGIN GENERATED\n")
    end = source.index("# END GENERATED")
    generated = "".join(_render(name, compile_proto(file)) for name, file in PROTO_FILES.items())
    path.write_text(source[:start] + generated + source[end:])


if __name__ == "__main__":
    regenerate()
