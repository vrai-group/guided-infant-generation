class Model_Template:

    def __init__(self):
        pass

    def build_model(self):
        pass

    def loss(self):
        pass

    def optimizer(self):
        pass

    def step_decay(self):
        pass

    def process_image(self, image, mean_pixel, norm):
        return (image - mean_pixel) / norm

    def unprocess_image(self, image, mean_pixel, norm):
        return image * norm + mean_pixel