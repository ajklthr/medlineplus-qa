from setuptools import setup, find_packages

# Read the requirements.txt file for dependencies
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='your_project_name',             # Replace with your project name
    version='0.1',                        # Version of your project
    packages=find_packages('src'),        # Tells setuptools to look for packages in the 'src' directory
    package_dir={'': 'src'},              # Specifies that the package root is the 'src' directory
    install_requires=required,            # Automatically installs dependencies from requirements.txt
    author='Your Name',                   # Replace with your name
    author_email='your_email@example.com',# Replace with your email
    description='A short description of your project',
    url='https://github.com/your/repo',   # Replace with the URL to your project (e.g., GitHub repository)
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Or the license you're using
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Minimum Python version required
)
