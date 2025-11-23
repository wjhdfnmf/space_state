

# Preparing the working environment

```bash
pacman -Syu ffmpeg teensy-loader-cli

python -m venv rocket_sim
source rocket_sim/bin/activate

pip install numpy scipy matplotlib astropy skyfield pytest
pip install h5py ruff mypy black
```

# Space State Guidance Computer – PlatformIO + Teensyduino + FreeRTOS

Teensy 4.1 (i.MX RT1062) real-time guidance computer using PlatformIO, official Teensyduino core, and SMP FreeRTOS.  
High-speed USB CDC-ACM for simulated ship data ingest from host Python scripts.


## Programmming the Teensy 4.1

space_state/
├── src/
│   ├── flight/           ← your real guidance code (main.cpp here)
│   └── test/
│       ├── imu_test.cpp
│       ├── quaternion_test.cpp
│       └── bench.cpp
├── platformio.ini

```bash
# build
pio run -e guidance

# upload 
pio run -e guidance --target upload
pio run --target upload # default guidance
```