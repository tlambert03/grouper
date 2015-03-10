try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Group Optimization Algorithm',
    'author': 'Talley Lambert',
    'download_url': 'Where to download it.',
    'author_email': 'My email.',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['grouper'],
    'scripts': [],
    'name': 'projectname'
}

setup(**config)