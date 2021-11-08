class Model_Template:

    def __init__(self):
        self.model = self.build_model()
        self.opt = self.optimizer()

    def build_model(self):
        pass

    def optimizer(self):
        pass

    def step_decay(self):
        pass

    #TODO: me le ritrovo anche nel dataset devo averle in un posto unico
    def process_image(self, image, mean_pixel, norm):
        return (image - mean_pixel) / norm

    def unprocess_image(self, image, mean_pixel, norm):
        return image * norm + mean_pixel