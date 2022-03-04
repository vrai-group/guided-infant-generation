class Model_Template:

    def __init__(self):
        self.model = self._build_model()
        self.opt = self._optimizer()

    def _build_model(self):
        pass

    def _optimizer(self):
        pass

    def prediction(self, **kwargs):
        pass

    def step_decay(self):
        pass