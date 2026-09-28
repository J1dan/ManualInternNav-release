import io
import os
import re
import sys

import setuptools

ROOT_DIR = os.path.dirname(__file__)
SUPPORTED_PYTHON_VERSIONS = [(3, 10), (3, 11), (3, 12)]

if tuple(sys.version_info[:2]) not in SUPPORTED_PYTHON_VERSIONS:
    msg = (
        f'Detected Python version {".".join(map(str, sys.version_info[:2]))}, which is not supported. '
        f'Only Python {", ".join(".".join(map(str, v)) for v in SUPPORTED_PYTHON_VERSIONS)} are supported.'
    )
    raise RuntimeError(msg)


def parse_readme(readme: str) -> str:
    """Parse the README.md file to be pypi compatible."""
    # Replace the footnotes.
    readme = readme.replace('<!-- Footnote -->', '#')
    footnote_re = re.compile(r'\[\^([0-9]+)\]')
    readme = footnote_re.sub(r'<sup>[\1]</sup>', readme)

    # Remove the dark mode switcher
    mode_re = re.compile(
        r'<picture>[\n ]*<source media=.*>[\n ]*<img(.*)>[\n ]*</picture>',
        re.MULTILINE,
    )
    readme = mode_re.sub(r'<img\1>', readme)
    return readme


long_description = ''
readme_filepath = os.path.join(ROOT_DIR, 'README.md')
if os.path.exists(readme_filepath):
    long_description = io.open(readme_filepath, 'r', encoding='utf-8').read()
    long_description = parse_readme(long_description)

def read_requirements(filename: str):
    """Read a requirement file relative to this setup script."""
    path = os.path.join(ROOT_DIR, 'requirements', filename)
    with open(path, 'r', encoding='utf-8') as requirements_file:
        return [
            line.split('#', 1)[0].strip()
            for line in requirements_file
            if line.split('#', 1)[0].strip()
        ]


install_requires = read_requirements('core_requirements.txt')
model_requires = read_requirements('model_requirements.txt')
isaac_requires = read_requirements('isaac_requirements.txt')
n1_requires = read_requirements('internvla_n1.txt')

setuptools.setup(
    name='internnav',
    version='0.2.0',
    packages=setuptools.find_packages(include=['internnav', 'internnav.*']),
    author='Intern Robotics',
    author_email='embodiedai@pjlab.org.cn',
    license='MIT',
    description='InternNav: A benchmark evaluation framework for navigation tasks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.10,<3.13',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    install_requires=install_requires,
    include_package_data=True,
    extras_require={
        # envs
        "isaac": isaac_requires,
        "habitat": [],
        "demo": [
            "gradio==5.45",
            "hf-xet==1.1.5",
            "huggingface-hub==0.33.4",
        ],
        # models
        "internvla_n1": n1_requires,
        "baseline": model_requires,
        "model": model_requires,
    },
)
