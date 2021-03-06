from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setup(
  name = 'DarkNeurons',         
  packages = ['DarkNeurons'],  


  version = '1.3.11',    
  long_description=readme(),
  long_description_content_type="text/markdown",     


  license='MIT',       
  description = 'A Library for Easy Implementation of Deep learning Techniques',   
  author = 'Tushar Goel',
  author_email = 'tgoel219@gmail.com',     
  url = 'https://github.com/Tushar-ml/DarkNeuron',   


  download_url = 'https://github.com/Tushar-ml/DarkNeuron/archive/v1.3.11.tar.gz',    
    
  keywords = ['Keras','Object_Detection','CNN'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'wget','tensorflow >= 1.15.0','numpy >= 1.18.4','scipy >= 1.4.1','matplotlib >= 3.2.1',
          'keras >= 2.1.4', 'pandas>=0.23.0','opencv-python>=4.2.0.32','Pillow>=5.3.0','netron'
      ],
  classifiers=[
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3'
  ],
)

