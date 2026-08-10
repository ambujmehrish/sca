"""Evaluation registry. Lazy so the light SCA harnesses (eval_missing, eval_calibration) can
be imported and unit-tested without the caption-eval / wandb dependency stack."""


class _LazyRegistry(dict):
    def __getitem__(self, key):
        if key not in self:
            if key == 'evaluation_mm':
                from .evaluation_mm import evaluate_mm
                self['evaluation_mm'] = evaluate_mm
        return super().__getitem__(key)


evaluation_registry = _LazyRegistry()
