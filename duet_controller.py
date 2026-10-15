import requests
import time

class DuetController:
    def __init__(self, ip_address):
        """
        Initializes the connection to the Duet 2 board over Wi-Fi.
        Args:
            ip_address: The local IP address of the Duet board (e.g. '192.168.1.100')
        """
        self.ip = ip_address
        self.base_url = f"http://{self.ip}/rr_gcode"
        
        # Test connection by requesting status
        try:
            # We just send an empty G-Code or M400 to test connection
            self.send_gcode("M400")
            print(f"Successfully connected to Duet 2 at {self.ip}")
        except Exception as e:
            print(f"Warning: Could not connect to Duet 2 at {self.ip}. Error: {e}")

    def send_gcode(self, gcode_str):
        """
        Sends a raw G-Code string to the Duet board via HTTP GET request.
        """
        try:
            response = requests.get(self.base_url, params={'gcode': gcode_str}, timeout=5)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Failed to send G-Code '{gcode_str}': {e}")
            return None

    def home_all(self):
        """Homes all axes."""
        print("Homing all axes...")
        self.send_gcode("G28")

    def move_x_absolute(self, position_mm, speed_mm_min=480):
        """
        Moves the X axis to an absolute coordinate.
        """
        gcode = f"G90\nG1 X{position_mm:.3f} F{speed_mm_min}"
        print(f"Moving X-axis to absolute {position_mm}mm...")
        self.send_gcode(gcode)

    def get_position(self):
        """
        Reads the current X, Y, Z coordinates from the Duet board API.
        Returns a list [X, Y, Z] or None if it fails.
        """
        try:
            resp = requests.get(f"http://{self.ip}/rr_status", params={'type': 2}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if 'coords' in data and 'xyz' in data['coords']:
                    return data['coords']['xyz'] # Returns [X, Y, Z]
        except Exception as e:
            print(f"Failed to read position from Duet: {e}")
            
        return None

    def move_x_relative(self, distance_mm, speed_mm_min=600):
        """
        Moves the X axis by a relative distance.
        Args:
            distance_mm: Distance to move (positive or negative)
            speed_mm_min: Feedrate (speed) of the move
        """
        # G91 = Relative positioning
        # G1 X... F... = Move X at feedrate
        # G90 = Return to Absolute positioning (optional, but safe)
        gcode = f"G91\nG1 X{distance_mm:.3f} F{speed_mm_min}\nG90"
        
        print(f"Moving X-axis by {distance_mm}mm...")
        self.send_gcode(gcode)

    def wait_for_move(self):
        """
        Sends M400 (Wait for current moves to finish) to ensure the motor 
        has completely stopped before we take our next camera frame.
        """
        self.send_gcode("M400")

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with your Duet's actual IP address!
    DUET_IP = "192.168.137.243" 
    
    duet = DuetController(DUET_IP)
    
    # Example: Move X by +0.1mm microstep, then wait for it to finish
    # duet.move_x_relative(0.1)
    # duet.wait_for_move()
