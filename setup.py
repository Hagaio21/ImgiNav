from setuptools import setup, find_packages

# This configuration tells pip how to install your package correctly.
setup(
    # The name of your package
    name="layout_generator",
    
    # The version of your package
    version="0.1.0",
    
    # find_packages tells pip to automatically find all packages in the 'src' directory
    packages=find_packages(where="src"),
    
    # This tells pip that the root of the package is the 'src' directory
    package_dir={"": "src"},
)
