import pymic
from . import RUNTIME

def get_device_id():
    return RUNTIME.get_device()

class Device(object):
    """Object represents a MIC device.
    """

    def __init__(self, device=None):
        if device is None:
            self.id = config.RUNTIME.get_device()
            self.stream = RUNTIME.get_stream()
        else:
            self.id = int(device)
            self.stream = pymic.devices[self.id].get_default_stream()

        self.__device_stack = []

    def __int__(self):
        return self.id

    def __enter__(self):
        idx = RUNTIME.get_device()
        self.__device_stack.append(idx)
        if self.id != idx:
            self.use()
        return self

    def __exit__(self, *args):
        RUNTIME.set_device(self.__device_stack.pop())

    def __repr__(self):
        return '<MIC Device %d>' % self.id

    def use(self):
        RUNTIME.set_device(self)

    def synchronize(self):
        pass

    @property
    def compute_capability(self):
        #TODO(superbo)
        return 'Hello'
