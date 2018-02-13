import torch
import pyro
import pyro.distributions as dist

import ast
import astor

model = r"""
alpha0 = Variable(torch.Tensor([10.0]))
beta0 = Variable(torch.Tensor([10.0]))
fairness = pyro.sample("latent", dist.Beta(alpha0, beta0))
f2 = sample("latent2", dist.Beta(alpha0, beta0))
fairness = pyro.sample("latent", dist.Beta(alpha0, beta0), obs=yo)
f2 = sample("latent2", dist.Beta(alpha0, beta0), obs=None)
pyro.observe("obs", dist.Bernoulli(fairness), data)
pyro.observe("obs", Bernoulli(fairness), data)
d = Bernoulli(fairness)
pyro.observe("obs", d, data)
"""

def is_obs_keyword(k):
    if k.arg != 'obs':
        return False
    if not isinstance(k.value, ast.NameConstant):
        return True
    if k.value.value is None:
        return False
    return True

class RemoveObserves(ast.NodeTransformer):

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id=="observe":
                return None
            elif node.func.id=="sample":
                if any( [is_obs_keyword(k) for k in node.keywords] ):
                    return None
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.attr=="observe":
                    return None
                elif node.func.attr=="sample":
                    if any( [is_obs_keyword(k) for k in node.keywords] ):
                        return None
        return node

class RemoveEmpties(ast.NodeTransformer):

    def visit_Expr(self, node):
        if hasattr(node, 'value'):
            return node
        else:
            return None

    def visit_Assign(self, node):
        if hasattr(node, 'value'):
            return node
        else:
            return None

print("model:\n", model, end='\n')

tree = ast.parse(model)
if 0:
    print("Original Model AST:\n")
    print(astor.dump(tree), end='\n')

tree = RemoveObserves().visit(tree)
tree = RemoveEmpties().visit(tree)
if 0:
    print("Modified Model AST:\n")
    print(astor.dump(tree), end='\n')

print("\nModified AST to Code:\n", astor.to_source(tree))
