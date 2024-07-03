import serial


class SerialCommunication:
    def __init__(self):
        self.com = serial.Serial("COM6", 115200, write_timeout=10)

    def sending_data(self, command: str) -> None:
        self.com.write(command.encode('ascii'))
