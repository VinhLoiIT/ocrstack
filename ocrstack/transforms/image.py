from PIL import Image
import torchvision.transforms as transforms

__all__ = [
    'ScaleHeight',
    'ImageTransform'
]


class ScaleHeight(object):
    def __init__(self, height: int):
        self.height = height

    def __call__(self, image: Image.Image):
        w, h = image.size
        factor = self.height / h
        w, h = int(w * factor), int(h * factor)
        image = image.resize((w, h))
        return image


class ImageTransform(object):
    def __init__(self, **kwargs):
        self.train = self.__train(**kwargs)
        self.test = self.__test(**kwargs)

    def __train(self, **kwargs):
        transform = transforms.Compose([
            ScaleHeight(kwargs['scale_height']),
            transforms.RandomApply([
                transforms.RandomAffine(0, None, None,
                                        shear=kwargs.get('shear', None),
                                        fillcolor=kwargs.get('fillcolor', 255)),
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(kwargs['mean'], kwargs['std']),
        ])
        return transform

    def __test(self, **kwargs):
        transform = transforms.Compose([
            ScaleHeight(kwargs['scale_height']),
            transforms.ToTensor(),
            transforms.Normalize(kwargs['mean'], kwargs['std']),
        ])
        return transform
