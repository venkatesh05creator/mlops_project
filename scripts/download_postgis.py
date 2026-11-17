import os
import sys
import urllib.request

DEST = os.path.join("runtime", "downloads", "postgis-bundle.zip")
URL = "http://download.osgeo.org/postgis/windows/pg16/postgis-bundle-pg16-3.6.2x64.zip"

os.makedirs(os.path.dirname(DEST), exist_ok=True)
print(f"Downloading PostGIS from {URL} to {DEST}...")

headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(URL, headers=headers)

with urllib.request.urlopen(req, timeout=300) as resp, open(DEST, 'wb') as out_file:
    total_size = int(resp.getheader('Content-Length', 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1MB
    while True:
        chunk = resp.read(chunk_size)
        if not chunk:
            break
        out_file.write(chunk)
        downloaded += len(chunk)
        if total_size > 0:
            percent = (downloaded / total_size) * 100
            sys.stdout.write(f"\rDownloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)")
        else:
            sys.stdout.write(f"\rDownloaded {downloaded / (1024*1024):.1f} MB")
        sys.stdout.flush()

print("\nDownload completed successfully!")
