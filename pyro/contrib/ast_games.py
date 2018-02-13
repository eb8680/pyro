import torch
import pyro
import pyro.distributions as dist
from torch.distributions import Beta, Bernoulli, Gamma
import ast
import astor

model = r"""
alpha0 = Variable(torch.Tensor([10.0]))
beta0 = Variable(torch.Tensor([10.0]))
fairness = pyro.sample("latent", dist.Beta(blah))
f2 = sample("latent2", dist.Gamma(alpha0, beta0))
fairness = pyro.sample("latent", dist.Beta(alpha0, beta0), obs=data)
f3 = pyro.sample("latent3", Gamma(alpha0, beta0), obs=None)
pyro.observe("obs1", dist.Bernoulli(fairness), data)
observe("obs2", Bernoulli(fairness), data)
"""

def is_obs_keyword(k):
    if k.arg != 'obs':
        return False
    if not isinstance(k.value, ast.NameConstant):
        return True
    if k.value.value is None:
        return False
    return True

def is_name(x):
    return isinstance(x, ast.Name)

def is_attribute(x):
    return isinstance(x, ast.Attribute)

def get_class(name):
    return globals()[name]

def print_dist_info(dist, sample_name):
    params = get_class(dist).params.items()
    print("Found non-obs sample statement <<%s>> with distribution of type %s (has_rsample = %s)" % (sample_name,
          dist, get_class(dist).has_rsample))
    for p, c in params:
        print("Has parameter %s with constraint %s %f" % (p, type(c).__name__, c.lower_bound))

def print_sample_info(node):
    dist_arg = node.args[1]
    sample_name = node.args[0].s
    if is_attribute(dist_arg.func):
        if is_name(dist_arg.func.value):
            print_dist_info(dist_arg.func.attr, sample_name)
    elif is_name(dist_arg.func):
        dist = dist_arg.func.id
        print_dist_info(dist_arg.func.id, sample_name)

class RemoveObserves(ast.NodeTransformer):
    def visit_Call(self, node):
        if is_name(node.func):
            if node.func.id=="observe":
                return None
            elif node.func.id=="sample":
                if any([is_obs_keyword(k) for k in node.keywords]):
                    return None
                else:
                    print_sample_info(node)

        elif is_attribute(node.func):
            if is_name(node.func.value):
                if node.func.attr=="observe":
                    return None
                elif node.func.attr=="sample":
                    if any([is_obs_keyword(k) for k in node.keywords]):
                        return None
                    else:
                        print_sample_info(node)

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
    print("\nOriginal Model AST:\n")
    print(astor.dump(tree), end='\n')

tree = RemoveObserves().visit(tree)
tree = RemoveEmpties().visit(tree)
if 0:
    print("\nModified Model AST:\n")
    print(astor.dump(tree), end='\n')

print("\nModified AST to Code:\n\n%s" % astor.to_source(tree))
