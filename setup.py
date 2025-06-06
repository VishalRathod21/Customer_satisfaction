from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    # Remove any comments and empty lines
    requirements = [req for req in requirements if not req.startswith('#') and req.strip()]

setup(
    name="customer-satisfaction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.10,<3.11',
    author="Your Name",
    author_email="your.email@example.com",
    description="E-Commerce Customer Satisfaction Prediction System",
    url="https://github.com/VishalRathod21/Customer_satisfaction",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
