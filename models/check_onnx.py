import onnx

m = onnx.load("yolov5nu.onnx")
onnx.checker.check_model(m)
print("valid")
print("opset:", m.opset_import[0].version)
for i in m.graph.input:
    print("input:", i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
for o in m.graph.output:
    print("output:", o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])