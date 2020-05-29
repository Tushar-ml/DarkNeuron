""" This Package will deal with implementation of Automatic Deep Learning 

which can reduce the time and Complexity for non-technical users

to train their own netwroks with significant accuracies along with

Deployement with Tensorflow Serving 

Author: Tushar Goel

"""

class Deep_Stack:
    """ This will be the Parent class for Deep Learning Techniques
    
    developed in this Package: Object Detection, Image Segmentation, Classfication 
    
    of Images, Certain Predefined Models """
    
    def __init__(self,working_directory=None,output_directory=None):
        """
        This Function will take the Directory of Data to be used
        
        and output directory for corresponding results___
        
        """
        self.working_directory= working_directory       # Input Directory
        self.output_directory = output_directory        # Output Directory
        
        print('\t\t########################\n')
        print('\t\tWelcome To The DEEPSTACK\n')
        print('\t\t########################\n')
              
