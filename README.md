# ContentBaseImageRetrieval_Simple
Building a database system to store and search food photos based on CBIR (Content-Based Image Retrieval) technique with Python

Choose 'master' branch (not 'main') to view project.


The 'begin' folder and the 'venv' folder are the ones I use for testing and editing, so you don't need to worry about them.
The 'Image' folder contains all the images in the database and their background removed images.
The 'final' folder contains all the code files and stores the feature vectors for retrieving our images.



To install all required dependencies in this project:
  1. Open Terminal of this project in IDE.
  2. Run 'pip3 install -r requirements.txt'
  

To Run based on data available in the project:
  1. Open folder 'final', then open file cbir.py, select the image path you want to process and retrieve (path_image_query = 'your_image_path')
  2. Run cbir.py
  
  
To preprocess the images:
  (Note: After pre-processing, the old data that I trained will be lost)
  1. Open 'pretreatment_images.py'
  2. Select the folder path containing images you want to preprocess (path_image_original  = 'your_folder_path')
  3. Un-Commnents at main.
  4. Run 'pretreatment_images.py'
  
