import pymic

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
