from __future__ import annotations

import ctypes

import paddle

import paddlefx


def mlir_compiler(gm: paddlefx.GraphLayer, inputs):
    print("Custom Compiler from FX Graph to MLIR:")
    print("-------------------------------------------------------------------")
    gm.graph.print_tabular()

    module = importer(gm, inputs)
    module = lowering(module)
    compiled = compile_module(module)
    data = prepare_data(gm, inputs)
    func = load_lib(compiled, data)

    return func


def prepare_data(gm: paddlefx.GraphLayer, inputs):
    data = {"inputs": [], "outputs": []}
    data["outputs"].append(
        ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(gm.forward(*inputs).numpy()))
        )
    )
    for inp in inputs:
        if isinstance(inp, paddle.Tensor):
            data["inputs"].append(
                ctypes.pointer(
                    ctypes.pointer(get_ranked_memref_descriptor(inp.numpy()))
                )
            )
        else:
            raise NotImplementedError
    return data


def importer(gm: paddlefx.GraphLayer, inputs):
    # Initialize the symbol table.
    symbolTable = {}
    # Create a module and build the operations into the module.
    module = Module.create()
    with InsertionPoint(module.body):
        # Parse the arguments.
        arguments = []
        for arg in inputs:
            shapeList = list(arg.shape)
            f32 = F32Type.get()
            tensorArg = RankedTensorType.get(shapeList, f32)
            arguments.append(tensorArg)

        # Generate the function.
        @func.FuncOp.from_py_func(*arguments)
        def generated_func(*args):
            # Convert arguments tuple into a list.
            argsList = list(args)
            # Traverse the graph and generate IR.
            for node in gm.graph.nodes:
                codegen(node, symbolTable, argsList)
            return symbolTable.get("output")

        generated_func.func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()
    print("-------------------------------------------------------------------")
    print("Printing the symbol table ...")
    # for symbol, op in symbolTable.items():
    #     import rich
    #     print(symbol, ": ", op)

    print("-------------------------------------------------------------------")
    print("Printing the generated MLIR ...")
    # print(module)
    # assert symbolTable["output"].type.dump() == "tensor<1x3x224x224xf32>"
    # meta_info = {
    #     "inputs": []
    # }
    return module


def codegen(node: paddlefx.Node, symbolTable, argsList):
    if node.op == "placeholder":
        # Bind the placeholder with args.
        symbolTable[str(node.name)] = argsList[0]
        argsList.pop(0)
    if node.op == "call_function":
        # Parse a call_function operation.
        if node.target.__name__ == "add":
            # Generate add operation.
            input1 = symbolTable.get(str(node.args[0]))
            input2 = symbolTable.get(str(node.args[1]))
            op = arith.AddFOp(input1, input2)
            symbolTable[str(node.name)] = op.result

    if node.op == "output":
        # Generating return operation.
        ret = symbolTable.get(str(node.args[0]))
        symbolTable["output"] = ret


def lowering(module: Module):
    print("-------------------------------------------------------------------")
    print("Bufferizing the module ...")
    pm = PassManager('builtin.module')
    pm.add("func.func(tosa-to-linalg)")
    pm.add("func.func(tosa-to-tensor)")
    pm.add("empty-tensor-to-alloc-tensor")
    pm.add("convert-elementwise-to-linalg")
    pm.add("arith-bufferize")
    pm.add("func.func(linalg-bufferize)")
    pm.add("func.func(tensor-bufferize)")
    pm.add("func-bufferize")
    pm.run(module.operation)
    # print(module)
    print("-------------------------------------------------------------------")
    print("Lowering the module to LLVM dialect ...")

    pm.add("func.func(buffer-deallocation)")
    pm.add("func.func(convert-linalg-to-loops)")
    pm.add("convert-scf-to-cf")
    pm.add("convert-linalg-to-llvm")
    pm.add("convert-arith-to-llvm")
    pm.add("expand-strided-metadata")
    pm.add("finalize-memref-to-llvm")
    pm.add("convert-func-to-llvm")
    pm.add("reconcile-unrealized-casts")
    pm.run(module.operation)
    # print(module)
    # translate module

    return module


def compile_module(module: Module):
    engine = ExecutionEngine(module)
    return engine


def load_lib(engine: ExecutionEngine, data):
    def invoke(*args):
        engine.invoke("generated_func", *data["outputs"], *data["inputs"])
        print(data["outputs"])

    return invoke
