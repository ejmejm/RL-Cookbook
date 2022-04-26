import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rl-cookbook",
    version="0.0.1",
    author="Edan Meyer",
    author_email="N/A",
    description="A cookbook of Reinforcement Learning methods and utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ejmejm/RL-Representation",
    project_urls={
        "Bug Tracker": "https://github.com/ejmejm/RL-Representation/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)