# Create the test image generator
import struct, math, zlib

def write_png(filename, pixels, width, height):
    def chunk(name, data):
        c = name + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    raw = b''
    for r in range(height):
        raw += b'\x00' + bytes(pixels[r*width:(r+1)*width])
    compressed = zlib.compress(raw, 9)
    png  = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0))
    png += chunk(b'IDAT', compressed)
    png += chunk(b'IEND', b'')
    with open(filename, 'wb') as f:
        f.write(png)

w, h = 512, 512
cx, cy = w//2, h//2
pixels = []
for r in range(h):
    for c in range(w):
        block = ((r // 64) + (c // 64)) % 2
        dist  = math.sqrt((c-cx)**2 + (r-cy)**2)
        in_c  = dist < 180
        pixels.append(200 if (in_c and block) else 50 if in_c else 240 if block else 20)

write_png("test_input.png", pixels, w, h)
print("Generated test_input.png (512x512)")
