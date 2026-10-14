import time
from machine import Pin

# --- PIN DEFINITIONS (DRV8833 Inputs) ---
# Notes: yellow a- d33, purple a+ d32, green b- d26, brown b+ d25
coil_a1 = Pin(32, Pin.OUT)  # D32 -> Pad A+
coil_a2 = Pin(33, Pin.OUT)  # D33 -> Pad A-
coil_b1 = Pin(25, Pin.OUT)  # D25 -> Pad B+
coil_b2 = Pin(26, Pin.OUT)  # D26 -> Pad B-

# Full-Step 2-Phase Sequence Matrix (A+, A-, B+, B-)
STEP_SEQUENCE = (
    (1, 0, 1, 0),  # Phase 0: A+, B+
    (0, 1, 1, 0),  # Phase 1: A-, B+
    (0, 1, 0, 1),  # Phase 2: A-, B-
    (1, 0, 0, 1)   # Phase 3: A+, B-
)

current_phase = 0

def disable_coils():
    """Turns off all coils to prevent motor heating."""
    coil_a1.value(0)
    coil_a2.value(0)
    coil_b1.value(0)
    coil_b2.value(0)

def step_bipolar(steps, delay_ms=12):
    """Steps the motor forward (+) or backward (-)."""
    global current_phase
    if steps == 0:
        return
    direction = 1 if steps > 0 else -1
    total_steps = abs(steps)
    for _ in range(total_steps):
        current_phase = (current_phase + direction) % 4
        phase_pins = STEP_SEQUENCE[current_phase]
        coil_a1.value(phase_pins[0])
        coil_a2.value(phase_pins[1])
        coil_b1.value(phase_pins[2])
        coil_b2.value(phase_pins[3])
        time.sleep_ms(delay_ms)
    disable_coils()

# Initialize coils to off
disable_coils()

print("\n--- MicroPython DRV8833 Stepper Ready ---")
print("Enter step count (e.g., 50 or -50), or 'q' to quit:")

while True:
    try:
        user_input = input("Steps: ").strip()
        if user_input.lower() == 'q':
            print("Exiting.")
            break
        steps = int(user_input)
        print(f"Stepping {steps} steps...")
        step_bipolar(steps, delay_ms=12)
        print("Done. Coils disabled.")
    except ValueError:
        print("Please enter a valid integer (e.g. 50, -50).")
    except KeyboardInterrupt:
        disable_coils()
        print("\nStopped.")
        break
