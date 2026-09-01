import setuptools


setuptools.setup(
    name="novavision-image-stack",
    version="0.1.0",
    author="DigiNova",
    author_email="info@diginova.com.tr",
    description="Image Stack component for NovaVision",
    url="https://github.com/Abd2023/novavision-image-stack",
    license="Apache-2.0",
    install_requires=[
        "sdk",
        "numpy>=1.26,<3",
        "opencv-python-headless>=4.10,<5",
        "pydantic>=1,<3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=[
        "novavision.image_stack",
        "novavision.image_stack.executors",
        "novavision.image_stack.models",
        "novavision.image_stack.utils",
    ],
    package_dir={"novavision.image_stack": "src"},
    python_requires=">=3.8",
)
