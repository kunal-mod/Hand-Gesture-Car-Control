import serial

ser = serial.Serial("COM6", 9600, timeout=1)

while True:
    x = input("Enter: ")
    if x == 'F':
        ser.write(b'F')
    elif x=='B':
        ser.write(b'B')
    elif x == 'L':
        ser.write(b'L')
    elif x=='R':
        ser.write(b'R')
    elif x=='S':
        ser.write(b'S')