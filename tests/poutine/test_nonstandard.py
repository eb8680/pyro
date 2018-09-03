from __future__ import absolute_import, division, print_function

import numbers

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

import pyro.poutine.nonstandard as nonstandard


def test_provenance():

    @nonstandard.make_nonstandard
    def add(x, y):
        return x + y

    @nonstandard.make_nonstandard
    def mul(x, y):
        return x * y

    @nonstandard.ProvenanceMessenger()
    def blah():
        return add(mul(2, 3), 4)

    b = blah()
    assert isinstance(b, nonstandard.ProvenanceBox)
    assert isinstance(b.value, numbers.Number)
    assert b.value == 10


def test_nested_provenance():

    @nonstandard.make_nonstandard
    def add(x, y):
        return x + y

    @nonstandard.make_nonstandard
    def mul(x, y):
        return x * y

    @nonstandard.ProvenanceMessenger()
    @nonstandard.ProvenanceMessenger()
    def blah():
        return add(mul(2, 3), 4)

    b = blah()
    assert isinstance(b, nonstandard.ProvenanceBox)
    assert isinstance(b.value, nonstandard.ProvenanceBox)
    assert isinstance(b.value.value, numbers.Number)
    assert b.value.value == 10


def test_lazy():

    @nonstandard.make_nonstandard
    def add(x, y):
        return x + y

    @nonstandard.make_nonstandard
    def mul(x, y):
        return x * y

    @nonstandard.LazyMessenger()
    def blah():
        return add(mul(2, 3), 4)

    b = blah()
    assert isinstance(b, nonstandard.LazyBox)
    assert b._value is None
    assert b.value == 10


def test_lazy_provenance():

    @nonstandard.make_nonstandard
    def add(x, y):
        return x + y

    @nonstandard.make_nonstandard
    def mul(x, y):
        return x * y

    @nonstandard.LazyMessenger()
    @nonstandard.ProvenanceMessenger()
    def blah():
        return add(mul(2, 3), 4)

    b = blah()

    assert isinstance(b, nonstandard.ProvenanceBox)
    assert isinstance(b.value, nonstandard.LazyBox)
    assert b.value._value is None
    assert b.value.value == 10
