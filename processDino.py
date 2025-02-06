import subprocess

scripts = [
    "models/DinoV2/generateEmbeddings.py",
    "models/DinoV2/faissInit.py",
]

# Ensure scripts have the correct shebang line
shebang = "#!/usr/bin/env python3\n"
for script in scripts:
    with open(script, 'r+') as f:
        content = f.read()
        if not content.startswith(shebang):
            f.seek(0, 0)
            f.write(shebang + content)

# Make scripts executable
for script in scripts:
    subprocess.run(['chmod', 'a+x', script])

# Run scripts and capture output
for script in scripts:
    result = subprocess.run(['python3', script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script}: {result.stderr}")
    else:
        print(f"Output of {script}: {result.stdout}")