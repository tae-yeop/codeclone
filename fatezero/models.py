

@dataclass
class UNetPseudo3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor

class UNetPseudo3DConditionModel(ModelMixin, ConfigMixin):

    def forward(self, )
    @classmethod
    def from_2d_model(cls, model_path, model_config):
        config_path = os.path.join(model_config, 'config.json')
        