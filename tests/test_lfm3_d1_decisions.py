"""The compatibility bridge preserves native answers and existing HTTP support."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from local_llm_server import _mlx_vlm_shims as shims


@pytest.fixture
def backend(monkeypatch):
    for name in ("mlx_vlm", "mlx_vlm.models"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    module = ModuleType("mlx_vlm.models.d1_lfm3_vl")
    utils = ModuleType("mlx_vlm.utils")
    decoded, attached, calls = [], [], []
    answer = {"answers": {"observation": {
        "type": "choice", "choice": "R3", "confidence": .7,
        "probabilities": {"R2": .3, "R3": .7}}}, "usage": {"input_tokens": 19}}

    class Model:
        def attach_processor(self, processor):
            attached.append(processor)
            self._engine = SimpleNamespace(processor=processor)

        def system_one(self, state, questions, *, media):
            calls.append((state, questions, media))
            return answer

    def load_image(value):
        image = object()
        decoded.append((value, image))
        return image

    module.Model = Model
    utils.load_image = load_image
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return SimpleNamespace(module=module, decoded=decoded, attached=attached,
                           calls=calls, answer=answer)


def test_http_bridge_decodes_images_and_preserves_native_answers(backend):
    shims._patch_d1_decisions()
    model, processor = backend.module.Model(), object()
    questions = {"observation": {"type": "choice", "criteria": {"R2": "R2", "R3": "R3"}}}
    data_url = "data:image/png;base64,fixture"
    result = model.predict(processor, None, questions, images=[data_url])
    assert result is backend.answer
    assert backend.decoded[0][0] == data_url
    assert backend.calls == [(None, questions, [backend.decoded[0][1]])]
    assert backend.attached == [processor]
    # A second request with the same processor keeps the native engine.
    model.predict(processor, "state", questions)
    assert backend.attached == [processor]
    assert backend.calls[-1] == ("state", questions, None)
    # Reapplying the shim must not wrap/replace predict again.
    predict = backend.module.Model.predict
    shims._patch_d1_decisions()
    assert backend.module.Model.predict is predict


def test_native_http_support_is_not_replaced(backend):
    model = backend.module.Model
    native_predict = object()
    model.decision_types = ("choice",)
    model.predict = native_predict
    assert "native HTTP decision support present" in shims._patch_d1_decisions()
    assert model.predict is native_predict
    assert model.decision_types == ("choice",)


def test_absent_d1_does_not_prevent_startup(backend, monkeypatch):
    monkeypatch.setitem(sys.modules, backend.module.__name__, None)
    assert "unavailable" in shims._patch_d1_decisions()


def test_unknown_model_api_is_not_adapted(backend):
    backend.module.Model = type("Model", (), {})
    assert "no native System One API" in shims._patch_d1_decisions()
    assert not hasattr(backend.module.Model, "predict")


def test_unsupported_media_is_rejected_before_native_inference(backend):
    shims._patch_d1_decisions()
    with pytest.raises(ValueError, match="image inputs only"):
        backend.module.Model().predict(object(), None, {}, videos=["video.mp4"])
    assert not backend.calls and not backend.decoded
