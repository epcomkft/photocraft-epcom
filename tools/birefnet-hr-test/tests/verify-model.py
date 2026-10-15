"""Optional real-model validation (not a full inference or GPU quality test).

Requires onnx 1.19.1, onnxruntime 1.23.2 and numpy. Arguments: source model.onnx,
adapted-fp32.onnx, produced by this tool's adaptPinnedModel implementation.
"""
import collections
import gc
import hashlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as H


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def inspect(path):
    onnx.checker.check_model(path)
    model = onnx.load(path)
    assert [(x.name, x.type.tensor_type.elem_type,
             [d.dim_value for d in x.type.tensor_type.shape.dim])
            for x in [*model.graph.input, *model.graph.output]] == [
                ('image', T.FLOAT, [1, 3, 2048, 2048]),
                ('alpha', T.FLOAT, [1, 1, 2048, 2048])]
    initializers = {}
    for weight in model.graph.initializer:
        assert weight.data_type == T.FLOAT
        assert np.isfinite(onnx.numpy_helper.to_array(weight)).all(), weight.name
        initializers[weight.name] = hashlib.sha256(weight.SerializeToString()).hexdigest()
    nodes = {n.name: (n.op_type, hashlib.sha256(n.SerializeToString()).hexdigest())
             for n in model.graph.node}
    assert len(nodes) == len(model.graph.node) == 29105
    assert next(n for n in model.graph.node if 'alpha' in n.output).op_type == 'Sigmoid'
    return initializers, nodes, {(o.domain, o.version) for o in model.opset_import}


assert digest(sys.argv[1]) == 'a548a1f3307250766bef50b174026b3cfa39a695bc1e42ec131c14fe82797ffd'
original = inspect(sys.argv[1]); gc.collect()
adapted = inspect(sys.argv[2]); gc.collect()
assert original[0] == adapted[0] and len(original[0]) == 873
changes = collections.Counter(op for name, (op, sha) in adapted[1].items()
                              if original[1][name][1] != sha)
assert changes == {'GridSample': 300, 'Gelu': 48}
assert original[2] == {('', 20)} and adapted[2] == {('', 19), ('com.microsoft', 1)}

# A small oracle verifies the two operator substitutions used by the adapter.
# It does NOT stand in for full-model inference on a user's GPU.
def graph(compat):
    nodes = [H.make_node('GridSample', ['x', 'grid'], ['sample'],
                        mode='bilinear' if compat else 'linear',
                        align_corners=0, padding_mode='zeros')]
    nodes.append(H.make_node('Gelu', ['sample'], ['y'],
                             **({'domain': 'com.microsoft'} if compat else {'approximate': 'none'})))
    model = H.make_model(H.make_graph(nodes, 'operator-equivalence', [
        H.make_tensor_value_info('x', T.FLOAT, [1, 1, 2, 2]),
        H.make_tensor_value_info('grid', T.FLOAT, [1, 2, 2, 2])],
        [H.make_tensor_value_info('y', T.FLOAT, [1, 1, 2, 2])]),
        opset_imports=[H.make_opsetid('', 19 if compat else 20)] +
        ([H.make_opsetid('com.microsoft', 1)] if compat else []), ir_version=9)
    onnx.checker.check_model(model)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(model.SerializeToString(), options,
                               providers=['CPUExecutionProvider'])

before, after = graph(False), graph(True)
feed = {'x': np.array([[-2., -.1], [.75, 4.]], dtype=np.float32).reshape(1, 1, 2, 2),
        'grid': np.array([[-.75, -.75], [.1, .8], [.8, .2], [1.2, 1.2]], dtype=np.float32).reshape(1, 2, 2, 2)}
a, b = before.run(None, feed)[0], after.run(None, feed)[0]
np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
assert np.isfinite(a).all() and np.isfinite(b).all()
print('PASS: source hash, ONNX checker, finite FP32 weights, identical 873 initializers,')
print('only 300 GridSample / 48 Gelu nodes changed; small CPU operator max error:', np.max(np.abs(a-b)))
