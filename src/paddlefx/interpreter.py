from __future__ import annotations

from typing import Any, Iterator

from .graph_layer import GraphLayer
from .node import Node, map_arg

__all__ = ["Interpreter"]


class Interpreter:
    def __init__(self, module: GraphLayer):
        assert isinstance(module, GraphLayer)
        self.module = module
        self.env: dict[Node, Any] = {}
        self.name = "Interpreter"

    def run(self, *args) -> Any:
        """Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order

        Returns:
            Any: The value returned from executing the Module
        """

        self.args_iter: Iterator[Any] = iter(args)

        for node in self.module.graph.nodes:
            try:
                self.env[node] = self.run_node(node)
            except Exception as e:
                msg = f"While executing {node}"
                msg = f"{e.args[0]}\n\n{msg}" if e.args else str(msg)
                e.args = (msg,) + e.args[1:]
                if isinstance(e, KeyError):
                    raise RuntimeError(*e.args) from e
                raise

            if node.op == "output":
                output_val = self.env[node]
                return output_val

    def run_node(self, n: Node) -> Any:
        """Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        # assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        return getattr(self, n.op)(n.target, args, kwargs)

    # Main Node running APIs
    def placeholder(self, target, args: tuple, kwargs: dict[str, Any]) -> Any:
        """Execute a ``placeholder`` node. Note that this is stateful:
        ``Interpreter`` maintains an internal iterator over
        arguments passed to ``run`` and this method returns
        next() on that iterator.

        Args:
            target (Target): The call target for this node.
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            Any: The argument value that was retrieved.
        """
        assert isinstance(target, str)
        if target.startswith("*"):
            # For a starred parameter e.g. `*args`, retrieve all
            # remaining values from the args list.
            return list(self.args_iter)
        else:
            try:
                return next(self.args_iter)
            except StopIteration as si:
                if len(args) > 0:
                    return args[0]
                else:
                    raise RuntimeError(
                        f"Expected positional argument for parameter {target}, but one was not passed in!"
                    ) from si

    def get_attr(self, target, args: tuple, kwargs: dict[str, Any]) -> Any:
        """Execute a ``get_attr`` node. Will retrieve an attribute
        value from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (Target): The call target for this node.
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The value of the attribute that was retrieved
        """
        assert isinstance(target, str)
        return self.fetch_attr(target)

    def call_function(self, target, args: tuple, kwargs: dict[str, Any]) -> Any:
        """Execute a ``call_function`` node and return the result.

        Args:
            target (Target): The call target for this node.
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the function invocation
        """
        assert not isinstance(target, str)

        # Execute the function and return the result
        return target(*args, **kwargs)

    def call_method(self, target, args: tuple, kwargs: dict[str, Any]) -> Any:
        """Execute a ``call_method`` node and return the result.

        Args:
            target (Target): The call target for this node.
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the method invocation
        """
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args

        # Execute the method and return the result
        assert isinstance(target, str)
        return getattr(self_obj, target)(*args_tail, **kwargs)

    def call_module(self, target, args: tuple, kwargs: dict[str, Any]) -> Any:
        """Execute a ``call_module`` node and return the result.

        Args:
            target (Target): The call target for this node.
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the module invocation
        """
        # Retrieve executed args and kwargs values from the environment

        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)

        return submod(*args, **kwargs)

    def output(self, target, args: tuple, kwargs: dict[str, Any]) -> Any:
        """Execute an ``output`` node. This really just retrieves
        the value referenced by the ``output`` node and returns it.

        Args:
            target (Target): The call target for this node.
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The return value referenced by the output node
        """
        return args[0]

    # Helper methods
    def fetch_attr(self, target: str):
        """Fetch an attribute from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (str): The fully-qualified name of the attribute to fetch

        Return:
            Any: The value of the attribute.
        """
        target_atoms = target.split(".")
        attr_itr = self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def fetch_args_kwargs_from_env(self, n: Node) -> tuple[tuple, dict]:
        """Fetch the concrete values of ``args`` and ``kwargs`` of node ``n``
        from the current execution environment.

        Args:
            n (Node): The node for which ``args`` and ``kwargs`` should be fetched.

        Return:
            Tuple[Tuple, Dict]: ``args`` and ``kwargs`` with concrete values for ``n``.
        """
        args = self.map_nodes_to_values(n.args, n)
        #        assert isinstance(args, tuple)
        kwargs = self.map_nodes_to_values(n.kwargs, n)
        assert isinstance(kwargs, dict)
        return args, kwargs

    def map_nodes_to_values(self, args, n: Node) -> Any:
        """Recursively descend through ``args`` and look up the concrete value
        for each ``Node`` in the current execution environment.

        Args:
            args (Argument): Data structure within which to look up concrete values

            n (Node): Node to which ``args`` belongs. This is only used for error reporting.
        """

        def load_arg(n_arg: Node) -> Any:
            if n_arg not in self.env:
                raise RuntimeError(
                    f"Node {n} referenced nonexistent value {n_arg}! Run Graph.lint() "
                    f"to diagnose such issues"
                )
            return self.env[n_arg]

        return map_arg(args, load_arg)
