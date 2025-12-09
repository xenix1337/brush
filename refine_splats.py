import sys
import struct
import re
from datetime import datetime

DUMP_DATA = False

def refine_splats():
    # Read all input from stdin (PLY data)
    data = sys.stdin.buffer.read()
    
    # Find header end
    header_end_idx = data.find(b"end_header\n")
    if header_end_idx == -1:
        sys.stderr.write("Error: Could not find end of header\n")
        sys.stdout.buffer.write(data)
        return

    header_bytes = data[:header_end_idx+11]
    body_bytes = data[header_end_idx+11:]
    header_str = header_bytes.decode('utf-8')
    
    # Parse header to find vertex count and property layout
    vertex_count = 0
    properties = []
    
    for line in header_str.split('\n'):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[2])
        elif line.startswith("property float"):
            properties.append(line.split()[2])
            
    if vertex_count == 0:
        sys.stderr.write("Error: No vertices found or count is 0\n")
        sys.stdout.buffer.write(data)
        return

    # Calculate stride and offsets
    bytes_per_float = 4
    stride = len(properties) * bytes_per_float
    
    try:
        f_dc_0_idx = properties.index("f_dc_0")
        f_dc_1_idx = properties.index("f_dc_1")
        f_dc_2_idx = properties.index("f_dc_2")
    except ValueError:
        sys.stderr.write("Error: Could not find f_dc properties\n")
        sys.stdout.buffer.write(data)
        return
    
    green_sh = 3.54
    
    new_body = bytearray(body_bytes)
    
    # Verify body size
    expected_size = vertex_count * stride
    if len(new_body) != expected_size:
        sys.stderr.write(f"Warning: Body size mismatch. Expected {expected_size}, got {len(new_body)}. Proceeding anyway.\n")
    
    for i in range(vertex_count):
        offset = i * stride
        
        # Pack new values
        # f_dc_0
        struct.pack_into("<f", new_body, offset + f_dc_0_idx * 4, 0.0)
        # f_dc_1
        struct.pack_into("<f", new_body, offset + f_dc_1_idx * 4, green_sh)
        # f_dc_2
        struct.pack_into("<f", new_body, offset + f_dc_2_idx * 4, 0.0)

    sys.stderr.write(f"Modified {vertex_count} splats to be green.\n")
    
    # Generate filename with timestamp up to microseconds
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"dumps/splats_green_{timestamp}.ply"
    
    # Dump data to file
    if DUMP_DATA:
        try:
            with open(filename, "wb") as f:
                f.write(header_bytes)
                f.write(new_body)
            sys.stderr.write(f"Dumped modified splats to {filename}\n")
        except Exception as e:
            sys.stderr.write(f"Failed to dump splats: {e}\n")

    # Write back
    sys.stdout.buffer.write(header_bytes)
    sys.stdout.buffer.write(new_body)
    sys.stdout.flush()

if __name__ == "__main__":
    refine_splats()
