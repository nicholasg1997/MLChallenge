import pytest


def test_default_config_is_the_full_fusion_model():
    from meld_emotion.training.config import TrainConfig
    c = TrainConfig()
    assert (c.use_text, c.use_faces, c.use_scene, c.use_track_id) == (True, True, True, True)
    assert c.context_k == 4 and c.max_text_tokens == 256
    assert c.d_model == 768 and c.max_frames == 32 and c.max_track_slots == 16
    assert c.modality_dropout == pytest.approx(0.15)


def test_every_ablation_preset_is_named_after_its_key():
    from meld_emotion.training.config import ABLATIONS
    assert set(ABLATIONS) == {"text_only_k0", "text_only_k4", "vision_only", "fusion",
                              "fusion_no_scene", "fusion_no_context", "fusion_no_trackid"}
    for key, cfg in ABLATIONS.items():
        assert cfg.name == key


def test_presets_flip_exactly_the_intended_switches():
    from meld_emotion.training.config import ABLATIONS
    t0 = ABLATIONS["text_only_k0"]
    assert (t0.use_text, t0.use_faces, t0.use_scene, t0.context_k) == (True, False, False, 0)
    assert ABLATIONS["text_only_k4"].context_k == 4 and not ABLATIONS["text_only_k4"].use_faces
    assert not ABLATIONS["vision_only"].use_text
    assert not ABLATIONS["fusion_no_scene"].use_scene
    assert ABLATIONS["fusion_no_context"].context_k == 0
    assert not ABLATIONS["fusion_no_trackid"].use_track_id


def test_config_for_applies_overrides_without_mutating_the_preset():
    from meld_emotion.training.config import ABLATIONS, config_for
    c = config_for("fusion", seed=3, epochs=1)
    assert (c.seed, c.epochs, c.name) == (3, 1, "fusion")
    assert ABLATIONS["fusion"].seed == 0


def test_config_for_rejects_unknown_preset():
    from meld_emotion.training.config import config_for
    with pytest.raises(KeyError):
        config_for("bogus")


def test_a_config_with_no_modality_is_rejected():
    from meld_emotion.training.config import TrainConfig
    with pytest.raises(ValueError):
        TrainConfig(use_text=False, use_faces=False, use_scene=False)


def test_to_dict_round_trips_through_constructor():
    from meld_emotion.training.config import TrainConfig
    c = TrainConfig(n_layers=4, seed=7)
    assert TrainConfig(**c.to_dict()) == c
