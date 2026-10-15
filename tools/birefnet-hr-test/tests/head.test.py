"""Regression: the actual JS adapter removes the >2 GiB head tensors while
preserving its FP32 output within floating-point roundoff. Uses small synthetic
feature maps, not a mocked inference. Requires onnx, onnxruntime, numpy, node.
"""
import copy
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as H, numpy_helper as N, TensorProto as T

root = '/model/decoder/'
low = root + 'decoder_block1/conv_out/Conv_output_0'
high = root + 'ipt_blk1/conv_out/Conv_output_0'
sizes = root + 'Concat_23_output_0'
weight = 'model.decoder.conv_out1.0.weight'
bias = 'model.decoder.conv_out1.0.bias'
resized = root + 'Resize_7_output_0'
concatenated = root + 'Concat_27_output_0'
logits = root + 'conv_out1/conv_out1.0/Conv_output_0'
conv_attrs = dict(dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1])
rng = np.random.default_rng(714)
w = rng.normal(0, .1, (1, 240, 1, 1)).astype(np.float32)
initializers = [N.from_array(w, weight), N.from_array(np.array([.13], np.float32), bias),
                N.from_array(np.zeros((1, 2, 2, 2), np.float32), 'fixture_grid')]
nodes = [
    H.make_node('Resize', [low, '', '', sizes], [resized], name=root+'Resize_7', coordinate_transformation_mode='align_corners', cubic_coeff_a=-.75, mode='linear', nearest_mode='floor'),
    H.make_node('Shape', [resized], ['shapeA'], name=root+'Shape_20'),
    H.make_node('Shape', [resized], ['shapeB'], name=root+'Shape_21'),
    H.make_node('Concat', [resized, high], [concatenated], name=root+'Concat_27', axis=1),
    H.make_node('Conv', [concatenated, weight, bias], [logits], name=root+'conv_out1/conv_out1.0/Conv', **conv_attrs),
    H.make_node('Sigmoid', [logits], ['alpha'], name='/Sigmoid'),
]
# Keep the same pinned export operator counts so the real fail-closed adapter
# runs unchanged on the small fixture. Unused nodes are pruned by ORT.
nodes += [H.make_node('GridSample', [low, 'fixture_grid'], [f'g{i}'], name=f'g{i}', mode='linear') for i in range(300)]
nodes += [H.make_node('Gelu', [low], [f'e{i}'], name=f'e{i}', approximate='none') for i in range(48)]
inputs = [H.make_tensor_value_info(low, T.FLOAT, [1,192,'lowH','lowW']),
          H.make_tensor_value_info(high, T.FLOAT, [1,48,'H','W']), H.make_tensor_value_info(sizes, T.INT64, [4])]
outputs = [H.make_tensor_value_info('alpha', T.FLOAT, [1,1,'H','W']),
           H.make_tensor_value_info(logits, T.FLOAT, [1,1,'H','W']),
           H.make_tensor_value_info('shapeA', T.INT64, [4]), H.make_tensor_value_info('shapeB', T.INT64, [4])]
model = H.make_model(H.make_graph(nodes, 'head-regression', inputs, outputs, initializers), opset_imports=[H.make_opsetid('',20)], ir_version=9)
adapter = (Path(__file__).resolve().parent.parent / 'onnx-compat.mjs').as_uri()
with tempfile.TemporaryDirectory() as temp:
    src, dst = Path(temp)/'before.onnx', Path(temp)/'after.onnx'
    onnx.save(model, src)
    js = 'import fs from "node:fs"; import {adaptPinnedModel} from '+json.dumps(adapter)+'; fs.writeFileSync(process.argv[2],adaptPinnedModel(fs.readFileSync(process.argv[1])));'
    subprocess.run(['node','--input-type=module','-e',js,str(src),str(dst)],check=True)
    converted = onnx.load(dst); onnx.checker.check_model(converted)
    assert all(resized not in n.input and resized not in n.output and concatenated not in n.input and concatenated not in n.output for n in converted.graph.node)
    weights = {t.name: N.to_array(t) for t in converted.graph.initializer}
    np.testing.assert_array_equal(weights['photocraft_hr_head_v3/low_weight'], w[:,:192])
    np.testing.assert_array_equal(weights['photocraft_hr_head_v3/high_weight'], w[:,192:])
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sessions = [ort.InferenceSession(str(p),options,providers=['CPUExecutionProvider']) for p in (src,dst)]
    max_alpha = max_logits = 0.
    for lh,lw,h,wid in [(5,7,17,23),(1,1,9,11),(2,3,1,1),(4,4,4,4)]:
        for magnitude in [1., 20.]:
            feed = {low:rng.normal(0,magnitude,(1,192,lh,lw)).astype(np.float32),
                    high:rng.normal(0,magnitude,(1,48,h,wid)).astype(np.float32),
                    sizes:np.array([1,192,h,wid],np.int64)}
            before, after = [s.run(None,feed) for s in sessions]
            assert all(np.isfinite(a).all() for a in after)
            np.testing.assert_allclose(before[0],after[0],rtol=5e-5,atol=4e-6)
            np.testing.assert_allclose(before[1],after[1],rtol=5e-5,atol=3e-5)
            np.testing.assert_array_equal(before[2],after[2]);np.testing.assert_array_equal(before[3],after[3])
            max_alpha=max(max_alpha,float(np.max(np.abs(before[0]-after[0]))))
            max_logits=max(max_logits,float(np.max(np.abs(before[1]-after[1]))))
    # Structural guard: a spatial 3x3 convolution must never be commuted.
    bad = copy.deepcopy(model)
    target = next(n for n in bad.graph.node if n.name==root+'conv_out1/conv_out1.0/Conv')
    a=next(a for a in target.attribute if a.name=='kernel_shape');a.ints[:]=[3,3]
    onnx.save(bad,src)
    failed=subprocess.run(['node','--input-type=module','-e',js,str(src),str(dst)],capture_output=True,text=True)
    assert failed.returncode != 0 and '1×1' in failed.stderr
    print('PASS: 8 real CPU head comparisons, shape outputs, weight slices, structural guard.')
    print('Maximum alpha difference:',max_alpha,'; logit difference:',max_logits)
