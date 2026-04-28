# src/config/pipeline_config.py
from pydantic import BaseModel

ABLATION_PRESETS: dict[str, dict[str, bool]] = {
    'full': {
        'use_quality_agent': True,
        'use_plan_validator': True,
        'use_memory': True,
        'use_explainability': True,
    },
    'no_validator': {
        'use_quality_agent': True,
        'use_plan_validator': False,
        'use_memory': True,
        'use_explainability': True,
    },
    'no_memory': {
        'use_quality_agent': True,
        'use_plan_validator': True,
        'use_memory': False,
        'use_explainability': True,
    },
    'no_quality_agent': {
        'use_quality_agent': False,
        'use_plan_validator': True,
        'use_memory': True,
        'use_explainability': True,
    },
}


class PipelineConfig(BaseModel):
    use_quality_agent: bool = True
    use_plan_validator: bool = True
    use_memory: bool = True
    use_explainability: bool = True

    @classmethod
    def from_ablation(cls, preset: str) -> 'PipelineConfig':
        if preset not in ABLATION_PRESETS:
            raise ValueError(
                f'Unknown ablation preset: "{preset}". '
                f'Valid options: {list(ABLATION_PRESETS.keys())}'
            )
        return cls(**ABLATION_PRESETS[preset])

    def ablation_label(self) -> str:
        """Human-readable label for logging and artifact naming."""
        for name, flags in ABLATION_PRESETS.items():
            if all(getattr(self, k) == v for k, v in flags.items()):
                return name
        return 'custom'