try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Group Optimization Algorithm',
    'author': 'Talley Lambert',
    'download_url': 'https://github.com/tlambert03/grouper.git',
    'author_email': 'tlambert03@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['grouper'],
    'scripts': [],
    'name': 'Grouper'
}

setup(**config)