import pytest

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from glom import T

from pyro.contrib.autoname import glom_name, scope


class B(object):
    def __init__(self):
        self.b = 3


def test_glom_to_str():

    expected_names = [
        "T.x",
        "T.y",
        "T.z[2]",
        "T.zz['a']",
        "T.ab.b",
        "T.zz['c'][0].b",
        "T.zz['c'][1]",
        "T.zz['c'][2]",
    ]

    glommed_names = list(map(str, [
        T.x,
        T.y,
        T.z[2],
        T.zz['a'],
        T.ab.b,
        T.zz['c'][0].b,
        T.zz['c'][1],
        T.zz['c'][2],
    ]))

    assert expected_names == glommed_names


def test_dynamic_simple_function():

    @glom_name
    def model():

        x = pyro.sample(dist.Bernoulli(0.5))
        y = pyro.sample(dist.Bernoulli(0.5))
        z = [x, y, None]
        i = 2
        z[i] = pyro.sample(dist.Bernoulli(0.5))
        zz = {}
        zz["a"] = pyro.sample(dist.Bernoulli(0.5))
        ab = B()
        ab.b = pyro.sample(dist.Bernoulli(0.5))
        zz["c"] = [B(), None, None, None]
        zz["c"][0].b = pyro.sample(dist.Bernoulli(0.5))
        for j in range(1, 3):
            zz["c"][j] = pyro.sample(dist.Bernoulli(0.5))

    expected_names = list(map(str, [
        T.x,
        T.y,
        T.z[2],
        T.zz['a'],
        T.ab.b,
        T.zz['c'][0].b,
        T.zz['c'][1],
        T.zz['c'][2],
    ]))

    tr = poutine.trace(model).get_trace()
    actual_names = [k for k, v in tr.nodes.items() if v["type"] == "sample"]
    print(tr.nodes)

    assert actual_names == expected_names


def test_dynamic_simple_class():

    class Model(object):

        @glom_name
        def model(self):

            x = pyro.sample(dist.Bernoulli(0.5))
            y = pyro.sample(dist.Bernoulli(0.5))
            z = [x, y, None]
            i = 2
            z[i] = pyro.sample(dist.Bernoulli(0.5))
            zz = {}
            zz["a"] = pyro.sample(dist.Bernoulli(0.5))
            ab = B()
            ab.b = pyro.sample(dist.Bernoulli(0.5))
            zz["c"] = [B(), None, None, None]
            zz["c"][0].b = pyro.sample(dist.Bernoulli(0.5))
            for j in range(1, 3):
                zz["c"][j] = pyro.sample(dist.Bernoulli(0.5))

    expected_names = list(map(str, [
        T.x,
        T.y,
        T.z[2],
        T.zz['a'],
        T.ab.b,
        T.zz['c'][0].b,
        T.zz['c'][1],
        T.zz['c'][2],
    ]))

    tr = poutine.trace(Model().model).get_trace()
    actual_names = [k for k, v in tr.nodes.items() if v["type"] == "sample"]
    print(tr.nodes)

    assert actual_names == expected_names


def test_glom_and_scope_simple_with():

    @glom_name
    def f1():
        with scope(prefix="f1"):
            x = pyro.sample(dist.Bernoulli(0.5))
            return x

    @glom_name
    def f2():
        f1()
        y = pyro.sample(dist.Bernoulli(0.5))
        return y

    tr1 = poutine.trace(f1, strict_names=False).get_trace()
    assert "f1/T.x" in tr1.nodes

    tr2 = poutine.trace(f2, strict_names=False).get_trace()
    assert "f1/T.x" in tr2.nodes
    assert "T.y" in tr2.nodes


@pytest.mark.xfail(reason="scope getting applied to f1 twice?")
def test_glom_and_scope_simple_decorator():

    @scope
    @glom_name
    def f1():
        x = pyro.sample(dist.Bernoulli(0.5))
        return x

    @glom_name
    def f2():
        f1()
        y = pyro.sample(dist.Bernoulli(0.5))
        return y

    tr1 = poutine.trace(f1, strict_names=False).get_trace()
    assert "f1/T.x" in tr1.nodes

    tr2 = poutine.trace(f2, strict_names=False).get_trace()
    assert "f1/T.x" in tr2.nodes
    assert "T.y" in tr2.nodes
