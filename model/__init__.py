"""Model registry. Lazy so that light SCA modules (centroid, losses_sca, prototypes, lora,
pmrl_loss) can be imported and unit-tested without pulling in the heavy encoder stacks."""


class _LazyRegistry(dict):
    def __getitem__(self, key):
        if key not in self:
            if key == 'gram':
                from .gram import GRAM
                self['gram'] = GRAM
            elif key == 'sca':
                from .sca import SCA
                self['sca'] = SCA
            elif key == 'gram_lora':
                from .baselines import GRAMLoRA
                self['gram_lora'] = GRAMLoRA
            elif key == 'pmrl':
                from .baselines import PMRL
                self['pmrl'] = PMRL
            elif key == 'gram_hyp':
                from .baselines import GRAMHyp
                self['gram_hyp'] = GRAMHyp
        return super().__getitem__(key)


model_registry = _LazyRegistry()
