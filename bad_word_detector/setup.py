from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="bad_word_detector",
    version="1.0.0",
    author="AI Assistant",
    author_email="user@example.com",
    description="Multilingual bad word detection system using AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/bad_word_detector",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-detector=scripts.train:main",
            "evaluate-detector=scripts.evaluate:main",
        ],
    },
)