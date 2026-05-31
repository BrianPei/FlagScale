import pickle
import sys
import types

import numpy as np
import pytest

try:
    import msgpack  # noqa: F401
except ModuleNotFoundError:
    fake_msgpack = types.ModuleType("msgpack")

    def _encode(value, default):
        converted = default(value) if default is not None else value
        if isinstance(converted, dict):
            return {key: _encode(item, default) for key, item in converted.items()}
        if isinstance(converted, (list, tuple)):
            return type(converted)(_encode(item, default) for item in converted)
        return converted

    def _decode(value, object_hook):
        if isinstance(value, dict):
            decoded = {key: _decode(item, object_hook) for key, item in value.items()}
            return object_hook(decoded) if object_hook is not None else decoded
        if isinstance(value, list):
            return [_decode(item, object_hook) for item in value]
        if isinstance(value, tuple):
            return tuple(_decode(item, object_hook) for item in value)
        return value

    def packb(value, default=None, **kwargs):
        return pickle.dumps(_encode(value, default))

    def unpackb(value, object_hook=None, **kwargs):
        return _decode(pickle.loads(value), object_hook)

    class Packer:
        def __init__(self, default=None, **kwargs):
            self.default = default

    class Unpacker:
        def __init__(self, object_hook=None, **kwargs):
            self.object_hook = object_hook

    fake_msgpack.packb = packb
    fake_msgpack.unpackb = unpackb
    fake_msgpack.Packer = Packer
    fake_msgpack.Unpacker = Unpacker
    sys.modules["msgpack"] = fake_msgpack

from flagscale.serve import msgpack_numpy


def test_pack_unpack_ndarray_roundtrip_preserves_dtype_and_shape():
    array = np.arange(12, dtype=np.float32).reshape(2, 3, 2)

    packed = msgpack_numpy.pack_array(array)
    unpacked = msgpack_numpy.unpack_array(packed)

    assert unpacked.dtype == array.dtype
    assert unpacked.shape == array.shape
    np.testing.assert_array_equal(unpacked, array)


def test_pack_unpack_numpy_scalar_roundtrip():
    scalar = np.int64(42)

    packed = msgpack_numpy.pack_array(scalar)
    unpacked = msgpack_numpy.unpack_array(packed)

    assert isinstance(unpacked, np.int64)
    assert unpacked == scalar


def test_msgpack_packb_unpackb_roundtrip_nested_numpy_values():
    payload = {
        "array": np.array([[1, 2], [3, 4]], dtype=np.int16),
        "scalar": np.float32(1.5),
        "plain": "value",
    }

    restored = msgpack_numpy.unpackb(msgpack_numpy.packb(payload), raw=False)

    np.testing.assert_array_equal(restored["array"], payload["array"])
    assert restored["scalar"] == payload["scalar"]
    assert restored["plain"] == "value"


@pytest.mark.parametrize(
    "value",
    [
        np.array([object()], dtype=object),
        np.array([1 + 2j], dtype=np.complex64),
        np.array([(1, 2)], dtype=[("x", "i4"), ("y", "i4")]),
    ],
)
def test_pack_array_rejects_unsupported_numpy_dtypes(value):
    with pytest.raises(ValueError, match="Unsupported dtype"):
        msgpack_numpy.pack_array(value)


def test_pack_and_unpack_passthrough_for_non_numpy_values():
    value = {"plain": [1, 2, 3]}

    assert msgpack_numpy.pack_array(value) is value
    assert msgpack_numpy.unpack_array(value) is value
