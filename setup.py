from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='LLM-Bridge',
    version='0.3.0',
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'httpx',
        'openai',
        'google-genai',
        'anthropic',
        'tiktoken',
        'pillow',
        'PyMuPDF',
        'python-docx',
        'openpyxl',
        'python-pptx',
        'pytest',
        'pytest-asyncio',
        'python-dotenv',
        'protobuf'
    ],
    author='windsnow1025',
    author_email='windsnow125@gmail.com',
    description='A Bridge for LLMs',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    python_requires='>=3.12',
)
