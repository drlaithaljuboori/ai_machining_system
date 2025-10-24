# scripts/install_kaolin.py
import subprocess
import sys
import os
import platform

def run_command(command, check=True):
    """Run a shell command and handle errors"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Success: {command}")
            return True
        else:
            print(f"✗ Failed: {command}")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed: {command}")
        print(f"Error: {e}")
        return False

def install_kaolin():
    """Install Kaolin with proper dependencies"""
    print("Installing Kaolin for AI Machining System...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor not in [8, 9, 10]:
        print("Warning: Kaolin works best with Python 3.8, 3.9, or 3.10")
    
    # Check CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA version: {cuda_version}")
    except ImportError:
        print("PyTorch not installed. Installing PyTorch first...")
        run_command(f"{sys.executable} -m pip install torch torchvision torchaudio")
    
    # Install Kaolin based on CUDA availability
    if cuda_available:
        print("Installing Kaolin with CUDA support...")
        # Try different installation methods
        kaolin_commands = [
            f"{sys.executable} -m pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{torch.__version__}/index.html",
            f"{sys.executable} -m pip install kaolin==0.15.0",
            f"{sys.executable} -m pip install git+https://github.com/NVIDIAGameWorks/kaolin"
        ]
    else:
        print("Installing Kaolin for CPU...")
        kaolin_commands = [
            f"{sys.executable} -m pip install kaolin==0.15.0",
            f"{sys.executable} -m pip install git+https://github.com/NVIDIAGameWorks/kaolin"
        ]
    
    # Try installation commands until one works
    kaolin_installed = False
    for command in kaolin_commands:
        if run_command(command, check=False):
            kaolin_installed = True
            break
    
    if not kaolin_installed:
        print("Failed to install Kaolin. Please install manually.")
        return False
    
    # Install other dependencies
    print("Installing other dependencies...")
    dependencies = [
        "trimesh", "pyvista", "open3d", "kaleido",
        "plotly", "matplotlib", "scikit-learn", "pandas",
        "reportlab", "openpyxl", "pyyaml"
    ]
    
    for dep in dependencies:
        run_command(f"{sys.executable} -m pip install {dep}")
    
    # Test installation
    print("Testing Kaolin installation...")
    try:
        import kaolin as kal
        print(f"✓ Kaolin imported successfully: version {kal.__version__}")
        
        # Test basic functionality
        import torch
        vertices = torch.randn(4, 3)
        faces = torch.tensor([[0, 1, 2], [1, 2, 3]])
        
        import kaolin.ops.mesh as mesh_ops
        normals = mesh_ops.face_normals(vertices.unsqueeze(0), faces)
        print("✓ Kaolin mesh operations working")
        
        return True
        
    except Exception as e:
        print(f"✗ Kaolin test failed: {e}")
        return False

if __name__ == "__main__":
    success = install_kaolin()
    if success:
        print("\n🎉 Kaolin installation completed successfully!")
        print("You can now run the AI Machining System.")
    else:
        print("\n❌ Kaolin installation failed.")
        print("Please check the errors above and install manually.")
        sys.exit(1)
