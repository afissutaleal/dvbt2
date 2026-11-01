import shutil
import os

src = r"C:\Users\user\.gemini\antigravity\scratch\dvbt2-propagation-ai"
dest = r"C:\Users\user\Downloads\dvbt2-propagation-ai.zip"
temp_base = r"C:\Users\user\Downloads\dvbt2-propagation-ai"

if os.path.exists(dest):
    os.remove(dest)

shutil.make_archive(temp_base, 'zip', src)
print(f"Zip updated successfully: {os.path.getsize(dest):,} bytes")
