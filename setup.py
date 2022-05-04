from setuptools import setup


with open("README.md", 'r') as f:
    long_description = f.read()

with open("requirements.txt", 'r') as f: 
    requirements = f.read().split()

setup(
	name="med3d", 
	version="0.1", 
	long_description=long_description,
	description="Basline code for AI projects", 
	author="Keno Bressem", 
	author_email="kenobressem@gmail.com", 
	packages=['med3d'],

)	
