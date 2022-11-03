import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyEngine",
    version="0.0.1",
    author="shuai.he",
    author_email="heshuai.sec@gmail.com",
    description="python inference engines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mightycatty/pyEngine",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)