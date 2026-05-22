from types import SimpleNamespace

from mflux.models.common.training.trainer import TrainingTrainer


class _DummyOptimizer:
    def __init__(self, state):
        self.optimizer = SimpleNamespace(state=state)
        self.saved_paths = []

    def save(self, path):
        self.saved_paths.append(path)


class _DummyTransformer:
    def __init__(self, supports_gc=True):
        self._gradient_checkpointing = False
        self._supports_gc = supports_gc

    def enable_gradient_checkpointing(self, enabled=True):
        self._gradient_checkpointing = enabled


class _DummyTransformerNoGC:
    pass


class TestGradientCheckpointing:
    def test_enabled_when_low_ram(self):
        transformer = _DummyTransformer()
        training_spec = SimpleNamespace(low_ram=True)
        enable_gc = getattr(transformer, 'enable_gradient_checkpointing', None)
        if training_spec.low_ram and callable(enable_gc):
            enable_gc(True)
        assert transformer._gradient_checkpointing is True

    def test_not_enabled_when_not_low_ram(self):
        transformer = _DummyTransformer()
        training_spec = SimpleNamespace(low_ram=False)
        enable_gc = getattr(transformer, 'enable_gradient_checkpointing', None)
        if training_spec.low_ram and callable(enable_gc):
            enable_gc(True)
        assert transformer._gradient_checkpointing is False

    def test_not_enabled_when_method_absent(self):
        transformer = _DummyTransformerNoGC()
        training_spec = SimpleNamespace(low_ram=True)
        enable_gc = getattr(transformer, 'enable_gradient_checkpointing', None)
        if training_spec.low_ram and callable(enable_gc):
            enable_gc(True)
        assert not hasattr(transformer, '_gradient_checkpointing')


class TestTrainingTrainer:
    def test_generate_previews_with_optimizer_offload_low_ram(self, monkeypatch):
        dummy_optimizer = _DummyOptimizer(state=["original_state"])
        training_state = SimpleNamespace(optimizer=dummy_optimizer)
        training_spec = SimpleNamespace(low_ram=True)
        adapter = object()

        preview_state_snapshots = []
        clear_cache_calls = []
        gc_calls = []

        def fake_generate_previews(_adapter, _training_spec, _training_state):
            preview_state_snapshots.append(_training_state.optimizer.optimizer.state)

        monkeypatch.setattr(TrainingTrainer, "_generate_previews", fake_generate_previews)
        monkeypatch.setattr("mflux.models.common.training.trainer.mx.clear_cache", lambda: clear_cache_calls.append(1))
        monkeypatch.setattr("mflux.models.common.training.trainer.gc.collect", lambda: gc_calls.append(1))
        monkeypatch.setattr("mflux.models.common.training.trainer.mx.load", lambda _path: {"k": "v"})
        monkeypatch.setattr(
            "mflux.models.common.training.trainer.tree_unflatten",
            lambda items: ["restored", items],
        )

        TrainingTrainer._generate_previews_with_optimizer_offload(adapter, training_spec, training_state)

        assert len(dummy_optimizer.saved_paths) == 1
        assert dummy_optimizer.saved_paths[0].name == "optimizer_offload.safetensors"
        assert preview_state_snapshots == [[]]
        assert dummy_optimizer.optimizer.state == ["restored", [("k", "v")]]
        assert len(clear_cache_calls) == 3
        assert len(gc_calls) == 3

    def test_generate_previews_with_optimizer_offload_non_low_ram(self, monkeypatch):
        dummy_optimizer = _DummyOptimizer(state=["original_state"])
        training_state = SimpleNamespace(optimizer=dummy_optimizer)
        training_spec = SimpleNamespace(low_ram=False)
        adapter = object()

        preview_state_snapshots = []
        clear_cache_calls = []
        gc_calls = []

        def fake_generate_previews(_adapter, _training_spec, _training_state):
            preview_state_snapshots.append(_training_state.optimizer.optimizer.state)

        monkeypatch.setattr(TrainingTrainer, "_generate_previews", fake_generate_previews)
        monkeypatch.setattr("mflux.models.common.training.trainer.mx.clear_cache", lambda: clear_cache_calls.append(1))
        monkeypatch.setattr("mflux.models.common.training.trainer.gc.collect", lambda: gc_calls.append(1))
        monkeypatch.setattr("mflux.models.common.training.trainer.mx.load", lambda _path: {"k": "v"})
        monkeypatch.setattr(
            "mflux.models.common.training.trainer.tree_unflatten",
            lambda items: ["restored", items],
        )

        TrainingTrainer._generate_previews_with_optimizer_offload(adapter, training_spec, training_state)

        assert len(dummy_optimizer.saved_paths) == 1
        assert dummy_optimizer.saved_paths[0].name == "optimizer_offload.safetensors"
        assert preview_state_snapshots == [[]]
        assert dummy_optimizer.optimizer.state == ["restored", [("k", "v")]]
        assert len(clear_cache_calls) == 3
        assert len(gc_calls) == 3
