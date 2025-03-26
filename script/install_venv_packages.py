import argparse
import os
import subprocess
import sys
from pathlib import Path


def install_packages_in_venv(venv_dir, python_version="3.12",
                             require_jupyter=False, http_proxy=None,
                             https_proxy=None):
    venv_dir = Path(venv_dir)
    python_executable = f"python{python_version}"

    if http_proxy:
        os.environ['http_proxy'] = http_proxy
    if https_proxy:
        os.environ['https_proxy'] = https_proxy

    try:
        subprocess.run([python_executable, "-m", "venv",
                       str(venv_dir)], check=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            f"Virtual environment creation failed: {e}\n"
            "Please ensure that the specified Python version is installed.\n"
        )
        sys.exit(1)

    pip_executable = venv_dir / "bin" / "pip"
    if not pip_executable.exists():
        pip_executable = venv_dir / "Scripts" / "pip"

    subprocess_args = [
        str(pip_executable),
        "install",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "statsmodels",
        "scipy",
        "pingouin",
        "sympy",
    ]

    if require_jupyter:
        subprocess_args.extend(["matplotlib_inline", "ipython"])

    process = subprocess.Popen(
        subprocess_args,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )

    process.communicate()
    if process.returncode != 0:
        sys.stderr.write("An error occurred during package installation.\n")
    else:
        sys.stdout.write("All packages have been successfully installed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv_dir", type=str, default="",
                        help="Path to the virtual environment directory")
    parser.add_argument("--require_jupyter", action="store_true",
                        help="Install Jupyter Notebook related packages")
    parser.add_argument("--python_version", default="3.12",
                        help=("Python version to be used for "
                              "creating the virtual environment (e.g., 3.12)"))
    parser.add_argument("--http_proxy", default="",
                        help=("HTTP proxy URL for pip "
                              "(e.g., http://proxy.example.com:port)"))
    parser.add_argument("--https_proxy", default="",
                        help=("HTTPS proxy URL for pip "
                              "(e.g., http://proxy.example.com:port)"))

    args = parser.parse_args()

    if args.venv_dir == "":
        default_dir = Path(__file__).parent.parent / "work" / "venv"
        print(
            "Virtual environment directory not specified. "
            f"Use default: {default_dir}")
        args.venv_dir = default_dir

    install_packages_in_venv(
        args.venv_dir,
        args.python_version,
        args.require_jupyter,
        args.http_proxy,
        args.https_proxy
    )
