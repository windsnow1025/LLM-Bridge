from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='LLM-Bridge',
    version='0.4.0',
    packages=find_packages(),
    install_requires=[
        'fastapi==0.115.9',
        'httpx==0.28.1',
        'openai==1.63.2',
        'google-genai==1.2.0',
        'anthropic==0.47.1',
        'PyMuPDF==1.25.3',
        'python-docx==1.1.2',
        'openpyxl==3.1.5',
        'python-pptx==1.0.2',
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
