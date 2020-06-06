from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setup(
  name = 'DarkNeuron',         
  packages = ['DarkNeuron'],  
<<<<<<< HEAD
  version = '1.2.1',    
  long_description=readme(),
  long_description_content_type="text/markdown",  
=======
  version = '1.2',      
>>>>>>> b972b8aa446424863d50ab68f5fbaabe83792471
  license='MIT',       
  description = 'A Library for Easy Implementation of Deep learning Techniques',   
  author = 'Tushar Goel',
  author_email = 'tgoel219@gmail.com',     
  url = 'https://github.com/Tushar-ml/DarkNeuron',   
<<<<<<< HEAD
  download_url = 'https://github.com/Tushar-ml/DarkNeuron/archive/v1.2.1.tar.gz',    
=======
  download_url = 'https://github.com/Tushar-ml/DarkNeuron/archive/v1.2.tar.gz',    
>>>>>>> b972b8aa446424863d50ab68f5fbaabe83792471
  keywords = ['Keras','Object_Detection','CNN'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'wget','tensorflow >= 1.15.0','numpy >= 1.18.4','scipy >= 1.4.1','matplotlib >= 3.2.1',
          'keras >= 2.1.4', 'pandas>=0.23.0','PyAutoGUI>=0.9.48','opencv-python>=4.2.0.32','Pillow>=5.3.0'
      ],
  classifiers=[
<<<<<<< HEAD
=======
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
>>>>>>> b972b8aa446424863d50ab68f5fbaabe83792471
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',
  ],
)

