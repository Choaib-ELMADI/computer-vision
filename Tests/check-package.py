import importlib.util

package = "opencv-python"

if (importlib.util.find_spec(package)) is None:
    print(f"{package} is not installed")
else:
    print(f"{package} is installed")
