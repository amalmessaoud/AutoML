# tests/test_pipeline_config.py
import pytest
from pydantic import ValidationError

from src.config.pipeline_config import ABLATION_PRESETS, PipelineConfig


class TestPipelineConfigDefaults:
    def test_all_flags_true_by_default(self):
        cfg = PipelineConfig()
        assert cfg.use_quality_agent is True
        assert cfg.use_plan_validator is True
        assert cfg.use_memory is True
        assert cfg.use_explainability is True

    def test_custom_flags(self):
        cfg = PipelineConfig(use_memory=False, use_explainability=False)
        assert cfg.use_memory is False
        assert cfg.use_explainability is False
        assert cfg.use_quality_agent is True
        assert cfg.use_plan_validator is True

    def test_invalid_flag_type_rejected(self):
        with pytest.raises(ValidationError):
            PipelineConfig(use_memory={'not': 'a bool'})


class TestFromAblation:
    def test_full_preset(self):
        cfg = PipelineConfig.from_ablation('full')
        assert cfg.use_quality_agent is True
        assert cfg.use_plan_validator is True
        assert cfg.use_memory is True
        assert cfg.use_explainability is True

    def test_no_validator_preset(self):
        cfg = PipelineConfig.from_ablation('no_validator')
        assert cfg.use_plan_validator is False
        assert cfg.use_quality_agent is True
        assert cfg.use_memory is True

    def test_no_memory_preset(self):
        cfg = PipelineConfig.from_ablation('no_memory')
        assert cfg.use_memory is False
        assert cfg.use_plan_validator is True

    def test_no_quality_agent_preset(self):
        cfg = PipelineConfig.from_ablation('no_quality_agent')
        assert cfg.use_quality_agent is False
        assert cfg.use_plan_validator is True
        assert cfg.use_memory is True

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match='Unknown ablation preset'):
            PipelineConfig.from_ablation('no_such_preset')

    def test_all_presets_are_valid(self):
        for preset in ABLATION_PRESETS:
            cfg = PipelineConfig.from_ablation(preset)
            assert isinstance(cfg, PipelineConfig)


class TestAblationLabel:
    def test_known_presets_return_name(self):
        for preset in ABLATION_PRESETS:
            cfg = PipelineConfig.from_ablation(preset)
            assert cfg.ablation_label() == preset

    def test_custom_config_returns_custom(self):
        cfg = PipelineConfig(
            use_quality_agent=False,
            use_plan_validator=False,
            use_memory=False,
            use_explainability=False,
        )
        assert cfg.ablation_label() == 'custom'