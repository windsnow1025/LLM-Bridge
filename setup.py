from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='LLM-Bridge',
    version='0.4.1',
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'httpx',
        'openai',
        'google-genai',
        'anthropic',
        'PyMuPDF',
        'python-docx',
        'openpyxl',
        'python-pptx',
    ],
    tests_require=[
        'pytest==8.3.4',
        'pytest-asyncio==0.25.3'
        'python-dotenv==1.0.1',
        'protobuf==5.29.3'
    ],
    author='windsnow1025',
    author_email='windsnow125@gmail.com',
    description='A Bridge for LLMs',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    python_requires='>=3.12',
)
