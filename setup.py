import setuptools

setuptools.setup(
    name='keras-retinanet',
    version='0.0.2',
    url='https://github.com/Xpitfire/keras-retinanet.git',
    author='Marius-Constantin Dinu',
    author_email='dinu.marius-constantin@hotmail.com',
    maintainer='Marius-Constantin Dinu',
    maintainer_email='dinu.marius-constantin@hotmail.com',
    packages=setuptools.find_packages(),
    install_requires=['keras', 'keras-resnet', 'six', 'scipy'],
    entry_points = {
        'console_scripts': [
            'retinanet-train=keras_retinanet.bin.train:main',
            'retinanet-evaluate-coco=keras_retinanet.bin.evaluate_coco:main',
            'retinanet-evaluate=keras_retinanet.bin.evaluate:main',
            'retinanet-debug=keras_retinanet.bin.debug:main',
        ],
    }
)
