#!/usr/bin/env python3
import serial
import time
import threading

# Auto-find Teensy port on Windows/Linux
def find_teensy():
    import serial.tools.list_ports
    for p in serial.tools.list_ports.comports():
        if "16C0:0483" in p.hwid or "Teensy" in p.description:
            return p.device
    raise OSError("Teensy not found — is it plugged in and programmed?")

port = find_teensy()
print(f"Connected to Teensy on {port}")

ser = serial.Serial(port, 2000000, timeout=1)

def reader():
    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"< {line}")
        time.sleep(0.001)

threading.Thread(target=reader, daemon=True).start()

print("Type messages — Teensy replies in UPPERCASE (Ctrl+C to quit)")
try:
    while True:
        msg = input()
        ser.write((msg + "\n").encode())
except KeyboardInterrupt:
    print("\nBye!")
    ser.close()