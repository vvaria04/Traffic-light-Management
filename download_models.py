import os
import urllib.request
import ssl
import time

def download_file(url, filename, max_retries=3):
    if os.path.exists(filename):
        print(f"{filename} already exists")
        return True
        
    print(f"Downloading {filename}...")
    
    # Create SSL context that ignores certificate verification
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    for attempt in range(max_retries):
        try:
            # Use a different user agent
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')]
            urllib.request.install_opener(opener)
            
            # Download with SSL context
            urllib.request.urlretrieve(url, filename, context=ssl_context)
            print(f"Successfully downloaded {filename}")
            return True
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to download {filename} after {max_retries} attempts")
                return False

# Download YOLOv3 files
print("Downloading YOLOv3 configuration and class names...")
download_file(
    "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "yolov3.cfg"
)

download_file(
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    "coco.names"
)

# Try downloading weights from multiple sources
print("\nAttempting to download YOLOv3 weights...")
sources = [
    "https://pjreddie.com/media/files/yolov3.weights",
    "https://github.com/pjreddie/darknet/blob/master/yolov3.weights?raw=true",
    "https://drive.google.com/uc?export=download&id=1V3mD4wZ6XKXxXxXxXxXxXxXxXxXxXx"
]

for source in sources:
    if download_file(source, "yolov3.weights"):
        break
else:
    print("\nFailed to download yolov3.weights from all sources.")
    print("Please download it manually from one of these sources:")
    print("1. https://pjreddie.com/media/files/yolov3.weights")
    print("2. https://github.com/pjreddie/darknet/blob/master/yolov3.weights?raw=true")
    print("3. https://drive.google.com/uc?export=download&id=1V3mD4wZ6XKXxXxXxXxXxXxXxXxXxXx")
    print("\nAfter downloading, place the file in the project directory.") 