import pymic


def check_available():
    return True if len(pymic.devices) > 0 else False


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


class Runtime(object):
    """Object keep state of pymic device and stream.
    """

    def __init__(self, deviceId=0):
        #TODO(superbo): finish initialize
        self.__id = int(deviceId)
        self.__lib = pymic.devices[self.__id].load_library("micpy.so")
        self.__stream = pymic.devices[self.__id].get_default_stream()

    def set_device(self, device):
        if isinstance(device, int):
            idx = device
            stream = pymic.devices[device].get_default_stream()
        elif isinstance(device, Device):
            idx = device.id
            stream = device.stream

        if self.__id == idx:
            return
        self.__id = idx
        self.__stream = stream

    def get_device(self):
        return self.__id

    def get_stream(self):
        return self.__stream

    def get_lib(self):
        return self.__lib

    def check_status(self):
        #TODO(superbo)
        return True


RUNTIME = Runtime(0)
