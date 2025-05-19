from setuptools import setup, find_packages

setup(
    name='lovmm',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    license=open('LICENSE').read(),
    zip_safe=False,
    description="Language-Conditioned Open-Vocabulary Mobile Manipulation with Pretrained Models",
    author='Shen Tan',
    author_email='shentan@stu.hit.edu.cn',
    install_requires=[line for line in open('requirements.txt').readlines() if "@" not in line],
)