import json
import base64

def get_base64_str_from_file(filepath):
    with open(filepath, "rb") as f:
        bytes_content = f.read() # bytes
        bytes_64 = base64.b64encode(bytes_content)
    return bytes_64.decode('utf-8') # bytes--->str  (remove `b`)

def save_base64_str_to_file(str_base64, to_file):
    bytes_64 = str_base64.encode('utf-8') # str---> bytes (add `b`)
    bytes_content = base64.decodebytes(bytes_64) # bytes
    with open(to_file, "wb") as f:
        f.write(bytes_content)

def test_base64():
    # image to/from base64
    image_path = "/home/mhbrt/Desktop/Wind/Multiscale/cod_logo.png"
    str_base64 = get_base64_str_from_file(image_path)
    save_base64_str_to_file(str_base64, "2.png")
    print("OK")

if __name__ == "__main__":
    test_base64()