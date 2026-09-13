def test_file_source_cuts_pending_audio_into_whole_30ms_frames_and_keeps_the_rest():
    from meld_emotion.inference.sources import FRAME_BYTES, FileSource
    src = FileSource.__new__(FileSource)          # no container: exercise the chunker alone
    src._pending = bytearray(b"a" * (2 * FRAME_BYTES + 100))
    chunks = src._chunks()
    assert [len(c) for c in chunks] == [FRAME_BYTES, FRAME_BYTES]
    assert len(src._pending) == 100
    assert src._chunks() == [] and len(src._pending) == 100
