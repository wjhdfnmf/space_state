import serial, struct, time

ser = serial.Serial('COM12', 2000000, timeout=1)  # find COM# in Device Manager

while True:
    t = time.time()
    pos = (1000, 2000, 3000)
    vel = (100, 50, 0)
    packet = struct.pack('<dffffff', t, *pos, *vel)  # little-endian doubles
    ser.write(packet)
    time.sleep(0.001)  # 1 kHz easy