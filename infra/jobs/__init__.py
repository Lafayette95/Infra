"""Jobs: operating models and strategies - read inputs, call the pure computation
(``infra.models``, ``infra.strategies``), write their stores. The only layer that writes model
and strategy outputs (``infra/models`` never writes storage). Scripts are thin wrappers over it;
nothing below imports it (``tests/test_architecture.py``)."""
