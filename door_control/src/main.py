# ---------------------------------------------------------------------------- #
#                                                                              #
# 	Module:       main.py                                                      #
# 	Author:       santi                                                        #
# 	Created:      4/28/2024, 2:55:10 PM                                        #
# 	Description:  EXP project                                                  #
#                                                                              #
# ---------------------------------------------------------------------------- #
#vex:disable=repl

# Library imports
from vex import *

# Brain should be defined by default
brain=Brain()

# Motor config
brain_inertial = Inertial()
door_motor = Motor(Ports.PORT1)


brain.screen.print("Door control!")

def serial_monitor():
    try:
        s = open('/dev/serial1','rb')
    except:
        raise Exception('serial port not available')
    
    while True:
        data= s.read(1)
        print(data)
        if data == b'a' or data == b'A':
            brain.screen.print_at("Open door", x=5, y=40)
            door_motor.spin_to_position(-90, DEGREES, 25, RPM)
        elif data == b'c' or data == b'C':
            brain.screen.print_at("Close door", x=5, y=40)
            door_motor.spin_to_position(0, DEGREES, 25, RPM)
            
        
t1=Thread(serial_monitor)


        
