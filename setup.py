from setuptools import setup, find_packages

setup(
    name="deeplazy",
    version="1.0.4",
    author="Eduardo de Moraes Froes",
    author_email="eduardomfroes@gmail.com",
    description="Lazy loading framework for large-scale LLMs using safetensors.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/deeplazy",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "tensorflow",
        "safetensors",
        "psutil",
        "rich",
        "redis"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
