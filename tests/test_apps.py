from torchapp.testing import TorchAppTestCase
from ecofuture.apps import EcoFuture


class TestEcoFuture(TorchAppTestCase):
    app_class = EcoFuture
