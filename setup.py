import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dl",
    version="0.0.1",
    author="yixin",
    author_email="zxcxzcz@qq.com",
    description="dl library and userful tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liuyixin-louis/dl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
