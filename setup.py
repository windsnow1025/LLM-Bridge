from setuptools import setup, find_packages

setup(
    name='LLM-Bridge',
    version='0.1.5',
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
    license='MIT',
    python_requires='>=3.12',
)