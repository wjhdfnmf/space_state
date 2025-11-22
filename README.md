

```bash
pacman -Syu ffmpeg

python -m venv rocket_sim
source rocket_sim/bin/activate

pip install numpy scipy matplotlib astropy skyfield pytest
pip install h5py ruff mypy black
```

# Space State Guidance Computer – PlatformIO + Teensyduino + FreeRTOS

Teensy 4.1 (i.MX RT1062) real-time guidance computer using PlatformIO, official Teensyduino core, and SMP FreeRTOS.  
High-speed USB CDC-ACM for simulated ship data ingest from host Python scripts.