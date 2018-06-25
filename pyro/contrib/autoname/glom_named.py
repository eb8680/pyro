import copy
from glom import T as _TTT  # noqa: F401
import astor
import gast
import inspect
import functools
import textwrap


class PrimitiveDetector(gast.NodeVisitor):
    """
    Checks whether a Call node contains ``pyro.sample`` or ``pyro.param``
    """
    def __init__(self):
        self._ret = False

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, gast.Attribute):
            self._ret = self._ret or \
                        isinstance(node.func.value, gast.Name) and \
                        node.func.value.id == "pyro" and \
                        node.func.attr in ("sample", "param")

    def visit(self, node):
        super(PrimitiveDetector, self).visit(node)
        return self._ret


class NameRewriter(gast.NodeTransformer):

    def _make_glom(self, node):
        new_node = copy.copy(node)
        new_node.ctx = gast.Load()
        ir_src = textwrap.dedent(astor.to_source(gast.gast_to_ast(
            gast.fix_missing_locations(new_node))))
        return gast.parse("str(_TTT." + ir_src + ")").body[0].value

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)

        def is_glom_decorator(d):
            if isinstance(d, gast.Name):
                return d.id == 'glom_name'
            elif isinstance(d, gast.Attribute):
                return is_glom_decorator(d.attr)
            return d == 'glom'
        node.decorator_list = list(filter(lambda d: not is_glom_decorator(d),
                                          node.decorator_list))
        return node

    def visit_Assign(self, node):
        if isinstance(node.value, gast.Call) and \
           PrimitiveDetector().visit(node.value) and \
           len(node.targets) == 1:
            new_name_node = self._make_glom(node.targets[0])
            if isinstance(node.value.args[0], gast.Str):
                node.value.args[0] = new_name_node
            else:
                node.value.args.insert(0, new_name_node)
        return node


def glom_name(fn):
    node = NameRewriter().visit(gast.parse(textwrap.dedent(inspect.getsource(fn))))
    fn.__globals__.update({"_TTT": _TTT})
    exec(astor.to_source(gast.gast_to_ast(gast.fix_missing_locations(node))),
         fn.__globals__)  # XXX gross...
    # return functools.wraps(fn)(fn.__globals__[fn.__code__.co_name])
    return fn.__globals__[fn.__code__.co_name]
