from setuptools import setup, find_packages

setup(
    name='diabetes_prediction',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'flask',
        'pandas',
        'numpy',
        'scikit-learn',
        'imbalanced-learn',
        'keras',
        'joblib',
        'matplotlib',
        'scipy',
        'tensorflow',
        'gunicorn',
    ],
    entry_points={
        'console_scripts': [
            'start-app = app:app.run',
        ],
    },
    author='Badreenath Gudipudi',
    author_email='badreenathgudipudi@gmail.com',
    description='A machine learning project for diabetes prediction using a classification model and regression analysis',
    long_description='This project includes preprocessing, model training, evaluation, and prediction for diabetes classification using a dataset of patient information.',
    long_description_content_type='text/markdown',
    url="https://github.com/badrinath0526/ML-Project"
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
