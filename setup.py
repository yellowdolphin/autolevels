from setuptools import setup

setup(
    name="autolevels",
    version="0.1",
    py_modules=["autolevels", "inference"],
    entry_points={
        'console_scripts': [
            'autolevels=autolevels:main',  # Adjust 'main' if your entry function has a different name
        ],
    },
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "pillow",
        "piexif"
    ],
)
