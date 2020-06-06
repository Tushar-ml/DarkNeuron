# Welcome to DarkNeuron !

**Dark Neuron**  will deal with implementation of Automatic Deep Learning  which can reduce the time and Complexity for non-technical users to train their own netwroks without Comprimising Accuracies for Classification of Images and Object Detection, most demanding tecniques for **Autonomous Systems** and **Medical Fields**.

> " By augmenting human performance, **AI** has the potential to markedly improve productivity, efficiency, workflow, accuracy and speed, both for physicians and for patients … What I’m most excited about is using the future to bring back the past: to restore the care in healthcare. " -  Eric Topol 

## CONTENTS :

	

 

 - <a  href='#classification'> Classification of Images </a>
	  
	 - <a href='#class_init'>Initialization</a>
	 - <a href='#class_prepare'>Preparation of Data</a>
	 -  <a href='#class_model'>Model Creation</a>
	 -  <a href='#class_train'>Training</a>
	 -  <a href='#class_prediction'>Prediction</a>
	 -  <a href='#class_visual'>Visualization of Prediction and Metrics</a>
	
 -  <a href='#object_detection'>Object Detection (YOLOv4)</a>
	 -  <a href='#obj_init'>Initialization</a>
	 -  <a href='#obj_prepare'>Preparation of Data</a>
	 -  <a href='#obj_train'>Model Training</a>
	 - <a href='#obj_detection'>Detection </a>
 -  <a href='#release'> Further Release </a>
 -  <a href='#license'> License </a>

# Installation 
		

    pip install DarkNeuron


## DarkNeuron Target Audience:
**DarkNeuron** is an Open Source Library . Target Audience  are:

 - All Experienced Data Scientists who wants to increase productivity with reduction in complexities and time.
 -  Professionals in Autonomous and Healthcare Industries to implement high accuracy models for the Production use.
 - Students of Data Science.

<h2 id='classification'>Classification of Images</h2>

**DarkNeuron Classification** has feature of implementing Pretrained Models on ImageNet Data. Users can directly  train pretrained models or can retrain their own models.  Models provided are:
				

 - InceptionV3
 - Xception
 - ResNet50
 - VGG16
 - VGG19
Further will be added on upcoming releases.

<h3 id='class_init'>Initialization of Classification Model</h3>

Initialization of Classification Model of DarkNeuron requires **working_directory**
as a main argument. It should have models and Raw_Data in it.
It can be Initialized as below:

    from DarkNeuron import Classify_Images
    classify = Classify_Images(working_directory = "Working Directory")
   
<h3 id='class_prepare'>Preparation of Data</h3>
Preparation of Data for Classification takes place in terms of whether the user wants to <b>Train the Model or Predict from the Model</b> and the <b>Method of Importing Images</b>:

 - **Directory** : To Import the whole folder with Images distributed in respective class folders.
 - **DataFrame** : Having a Dataframe containing Image filenames and corresponding Labels.
 - **Point** : To provide input as an array like :  X_train , Y_train .
 - **Image** : To provide Single Image as an Input (suggested for Prediction Phase)
<b>Let's See each of them with their necessary arguments</b>

<h4><u> Method: Directory</u></h4>
<b> Code Syntax:</b> (Continue from above....)

    train,val,labels = classify.Prepare_the_Data(method = 'directory', train =True,
				num_classes = 2, batch_size = 32, #Default
				target_image_size = (224,224,3) #Default
				model_name = 'InceptionV3',
				user_model = None, #Default,
				training_image_directory = 'Directory_Path
				validation_image_directory = None,
				)

Let's See each argument and their default values:

 - **train**: **False** (for Prediction) and **True** (Training)
 - **num_classes** : No of classes in user data     (Default: 2)
 - **batch_size** : Batch Size of Training Data      (Default:32)
 - **target_image_size** :  Image Input Size used for Creation and Preprocessing of the Input
 - **model_name**: Name of Pretrained Model , **not required when provide user_model name** (Default: None)(**Same for all Method**)
 - **user_model**: user pretrained model , input taken as an model means Load the model using `classify.load_model(user_model_name)(**Same for all Method**)
 - **training_image_directory**: Full Path of Training Image Directory (Must for Training )		(Default: None)
 - **validation_image_directory**: Full Path of Validation Image Directory (Default=None)
 - **test_image_directory**: Test Image Directory path, **only used when train = False**
 
<h4><u>Method: DataFrame</u></h4>
<b>Code Syntax:</b>

    train,val,labels = classify.Prepare_the_Data(method = 'dataframe', train = True,
				num_classes = 2,batch_size = 32,
				dataframe = df ,
				x_col_name = 'filename',
				y_col_name = 'label',
				image_directory = None,
				split = 0.1 )

Let's Understand the above arguments:

 - **dataframe**: Loaded Dataframe variable ( refer Pandas for defining DataFrame)
 - **x_col_name**: name of Image file names containing column name
 - **y_col_name**: name of Labels containing column name
 - **image_directory**: **Only required if x_col_name has relative path for images**
 - **split**: Spliting of data automatically for validation and Training puroses
 
 <h4><u>Method: Point</u></h4>
 <b>Code Example....</b>
 

    train,val,labels = classify.Prepare_the_Data(method = 'point',train = True,
				x_train = x_train,y_train = y_train,
				x_test = x_test,y_test = y_test)

Let's Understand each argument:

 - **x_train**:  Input X variable for Training
 - **y_train**: Target Y variable for Training
 - **x_test**: Input X variable for Testing and Validation
 - **y_test**: Target Y variable for Testing and Validation

<h4><u>Method: Image</u></h4>
<b>Code Syntax:</b>

    test = classify.Prepare_the_Data(method='image',train = False,
				image_path = 'Path of the Image',
				grayscale=False
				)

 - **image_path**: Path of the Image to predicted
 - **grayscale**: To load the image with grayscale feature

<h3 id='class_model'> Model Creation</h3>
This Feature takes no argument , but <b>necessary when user provide model_name</b>  .
<br>
It will create the full structure of the model  based on the data provided in Prepare the Data function call.

    model = classify.Create_the_Model()

That's it. Model will be created and Generated.
**If you have PreDownloaded weights, then must sure the following:**

 - Put the model in working_directory
 - If Training is **False**: (Names of the model to be save)
	 - InceptionV3 : 'inceptionv3_model.h5'
	 - ResNet50 : 'resnet50_model.h5'
	 - VGG16: 'vgg16_model.h5'
	 - VGG19: 'vgg19_model.h5'
	 - Xception: 'xception_model.h5'
  - If Training is **True**: (Names of the model to be save)
	 - InceptionV3 : 'inceptionv3_notop_model.h5'
	 - ResNet50 : 'resnet50_notop_model.h5'
	 - VGG16: 'vgg16_notop_model.h5'
	 - VGG19: 'vgg19_notop_model.h5'
	 - Xception: 'xception_notop_model.h5'

**Otherwise, it will automatically Download the weights.**

<h3 id='class_train'> Model Training</h3>

This Feature will be used for Model Training purposes .
<b> Code Syntax:</b>
	

    model = classify.Train_the_Model(model = model,
				    rebuild = False,
				    train_data_object = train,
				    validation_data_object = train,
				    epochs = 10,
				    optimizers = 'adam',
				    loss = 'binary_crossentropy',
				    fine_tuning = False,
				    layers = 20,
				    metrics = ['accuracy'],
				    validation_steps = 80,
				    steps_per_epoch = 50,
				    callbacks = None
				    )

 - **model**: model created from previous step.
 - **rebuild**: **only**, when model_name provided, set to **True**
 - **train_data_object**: generators get from Prepare the Data function
 - **epochs**: No of Steps for training
 - **optimizer**: Suitable Optimizer for model
 - **loss**: Loss function for model
 - **fine_tuning**: To do Fine_tuning or not 
 - **layers**: **Only required when Fine Tuning set to True**, number of layers from bottom to be trained or to train all layers provide **'all'** argument
 - **metrics**: To be provided as **list**.
 - **callbacks**: To be provided as **list** by the user for early_stopping or Checkpoint.

<h3 id='class_prediction'> Prediction </h3>
This Feature will be used for Prediction from the model on the Test Dataset.
<br><b>To do this step, First Prepare the Data with train argument set to False and obtain test object from it.</b>
<br><br><b>Code Syntax:</b>

    classify.Predict_from_the_Model(labels = labels,
					model = model,
					img = None,
					generator = None,
					top = 5
					)

 - **labels**: Labels provided as List or provided from generated labels in Prepare the data function during training.  (See Above)
 - **model**: Model generated by Training or due to loading user own model.
 - **img**: only if method: image
 - **generator**: Test Data Object generated from Prepare the Data Function call.
 - **top**: Top k predictions for image.

<h3 id='class_visual'>Visualization of Predictions and Metrics</h3>

<h4> Metrics Visualization</h4>

    classify.Visualize_the_Metrics()

<h4> Prediction Visualization</h4>

    classify.Visualize_the_Predictions(number = 20)

 - **number**: No of Images or Predictions to Visualize

<h3> Here Comes the Ending to Classification Part</h3>
<h3> Let's move on to Object Detection Part</h3>

<h2 id = 'object_detection'>Object Detection (YOLOv4)</h2>


<h3 id='obj_init'>Initialization of Object Detection Model</h3>
This Function will take working directory as an argument where the training data is present and weights be present . If no weights are there then it will be downloaded.<br>
If you have predefined yolov4 weights : Named it as --> 'yolov4.weights'
If you have predefined yolov4 model: Named it as --> 'yolov4.h5'

    from DarkNeuron import YOLOv4
    yolo = YOLOv4( working_directory )

<h3 id='obj_prepare'> Preparation of Data</h3>

For this Function, All Images and corressponding labels should be in working_Directory within no sub folder.( For Simplicity, Train directory = Working directory).
This Function take file in three formats and converted them into YOLO Format Automatically:

 - csv
 - xml
 - text files
 
 <b>Code Syntax:</b>
 

    yolo.Prepare_the_Data(file_type,
			    dataframe_name = None,
			    class_file_name = None
			    )

 - **file_type**: This contain file_type: whether csv, xml, or text_files
 - **dataframe_name**: This should be given as name of csv file in working_directory
 - **class_file_name**: provide name of the  class list as text file in working directory

<h3 id='obj_train'> Model Training</h3>
This Function will be used to Train the model on user custom data set.
<br>
There are two process involved :

 - Process_1 : Simple Training
 - Process_2 : After Process_1, Fine tuning (Highly Recommended)

<b>Code Syntax:</b>

    yolo.Train_the_Yolo(model_name = 'yolov4.h5',
			    input_shape = (608,608) #Multiple of 32 required
			    score = 0.5,
			    iou = 0.5,
			    epochs1 = 50, #For Process 1
			    epochs2 = 51, #For Process 2
			    batch_size1 = 32,
			    batch_size2 = 4,
			    validation_split = 0.1,
			    process1 = True,
			    process2 = True
			   )

 - **model_name**: If user have predefine model, can provide the name.
 - **input_shape**: Input Shape for the model .
 - **score**: Score Threshold.
 - **iou**: Intersection Over Union thresholf over training (must change for better accuracy)
 - **epochs1, epochs2**: Epochs for Different Processes described above.
 - **batch_size1, batch_size2**: Batch Size for Differn Purposes
 - **process1, process2**: Process to be Done (Default: True)

 <h3 id='obj_detection'> Detection </h3>
This Function will be used to detect objects from video and Images.
This Function has following features:<br>

 - **Real Time Detection** --> Play a Video on your laptop or Take real data from CCTV , and run the model , it will take your screen as an input and Detect objects in it.
 - **Web Cam Detection** --> It will Detect using webcams and can also be used by Mobile Phone Cameras ( see IPWebCam )
 - **Choose Class** --> You can choose your own prediction classes , means which object to predict which to not. For Example, on COCO dataset , it has 80 labels, then you should pass person to the function, it will detect only person, leave everthing else as it is.
 - **Tracking**: Multi Object Tracking with DeepSORT (Deep Simple Online Real Tracking)
 
 <b> Code Synatax:</b>
 

    yolo.Detect(test_folder_name = 'test',
		    model_name = None,
		    cam = False,
		    real_time = False,
		    videopath = 0,
		    classes = [],
		    score = 0.5,
		    tracking = False
		    )

 - **test_folder_name**: Test folder name in working directory ( images and video both, it will detect automatically and take actions according to it)
 - **model_name**: Model name saved in working_directory by Training, otherwise it will take yolov4.h5 by default.
 - **cam**: To enable Web Cam Detection
 - **real_time**: To enable Real Time Detection
 - **videopath**: Path to the video to detect
 - **classes**: Selelctive choosing of Classes for Detections (**Provide as List**)
 - **score**:  Threshold of Score for Prediction
 - **tracking**: DeepSort Tracking to be enable or not

<h2 id='release'> Further Release</h2>

 - [ ] Improvement in Tracking 
 - [ ] Artificial Neural Networks user friendly implementation
 - [ ] Visualization of Neural Networks
 
 <h2 id='license'>License</h2>
 MIT  License

**Copyright  (c)  2020  DarkNeuron  Tushar-ml**

Permission  is  hereby  granted,  free  of  charge,  to  any  person  obtaining  a  copy

of  this  software  and  associated  documentation  files  (the  "Software"),  to  deal

in  the  Software  without  restriction,  including  without  limitation  the  rights

to  use,  copy,  modify,  merge,  publish,  distribute,  sublicense,  and/or  sell

copies  of  the  Software,  and  to  permit  persons  to  whom  the  Software  is

furnished  to  do  so,  subject  to  the  following  conditions:

The  above  copyright  notice  and  this  permission  notice  shall  be  included  in  all

copies  or  substantial  portions  of  the  Software.

THE  SOFTWARE  IS  PROVIDED  "AS  IS",  WITHOUT  WARRANTY  OF  ANY  KIND,  EXPRESS  OR

IMPLIED,  INCLUDING  BUT  NOT  LIMITED  TO  THE  WARRANTIES  OF  MERCHANTABILITY,

FITNESS  FOR  A  PARTICULAR  PURPOSE  AND  NONINFRINGEMENT.  IN  NO  EVENT  SHALL  THE

AUTHORS  OR  COPYRIGHT  HOLDERS  BE  LIABLE  FOR  ANY  CLAIM,  DAMAGES  OR  OTHER

LIABILITY,  WHETHER  IN  AN  ACTION  OF  CONTRACT,  TORT  OR  OTHERWISE,  ARISING  FROM,

OUT  OF  OR  IN  CONNECTION  WITH  THE  SOFTWARE  OR  THE  USE  OR  OTHER  DEALINGS  IN  THE SOFTWARE.

 

