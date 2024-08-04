# Importing the os module to interact with the operating system,
# the csv module to handle CSV file operations,
# Importing necessary classes from the transformers library,
# Importing the Image class from the PIL library to handle image operations,
# Importing the torch library for PyTorch operations.
import os
import csv
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
import torch
import pandas as pd


# 1. Setting up the model and processor
#    Specifying the model ID, Allowing the loading of remote code,
#    Mapping the model to use CUDA (GPU), Automatically selecting the appropriate torch data type,
#    Finally, Setting the model to evaluation mode and moving it to CUDA.

model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id,  
                                            trust_remote_code=True,  
                                            cache_dir="my_models/Florence_2",  
                                            device_map="cuda",  
                                            torch_dtype='auto').eval().cuda()
                                           
processor = AutoProcessor.from_pretrained(model_id,  
                                          trust_remote_code=True)




# 2. Create function to run the model on a given image 
#    Setting the prompt for the model, Processing the input and moving to CUDA,
#    Generating text from the input IDs, Using the pixel values from the image,

def run_example(image):
    prompt = '<MORE_DETAILED_CAPTION>'
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,  # Setting the maximum number of new tokens to generate
      early_stopping=False, # Allowing the generation process to continue until all beams finish
      do_sample=False,      # Disabling sampling for deterministic output
      num_beams=3,          # Using beam search with 3 beams.
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,     # Post-processing the generated text
        task=prompt,        # Using the initial prompt for the task
        image_size=(image.width, image.height)  # Setting the image size for post-processing
    )
    # Returning the parsed detailed caption
    return parsed_answer['<MORE_DETAILED_CAPTION>']





# 3. Function to get the detailed caption for an image,
#    Opening and converting the image to RGB,
#    then Running the example function on the image.

def get_captune(image_path):
    image = Image.open(image_path).convert("RGB") 
    return run_example(image) 





# 4. Function to process all images in a directory, Listing all files in the directory,
#    Filtering and sorting image files.

def process_images(directory):
    files = os.listdir(directory) 
    image_files = sorted([f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))])
    
    # Opening a CSV file to write the results, defining the CSV field names, Finally, Creating a CSV writer object 
    with open('captune_data.csv', 'a', newline='') as csvfile:
        fieldnames = ['image_name', 'appartmentID', 'ImageID', 'Image_captune']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames) 
        
        # Writing the header if the file is empty.
        if csvfile.tell() == 0:
            writer.writeheader()
           
        # Printing the image name, then extracting the apartment ID and image ID,
        # Constructing the full path to the image, finally getting the detailed caption for the image.
        for image_name in image_files:
            print(image_name) 
            appartmentID, imageID = image_name.rsplit('_', 1)[0], image_name.rsplit('_', 1)[1].split('.')[0]  
            image_path = os.path.join(directory, image_name) 
            image_captune = get_captune(image_path)
            
            writer.writerow({'image_name': image_name,  
                             'appartmentID': appartmentID,  
                             'ImageID': imageID,  
                             'Image_captune': image_captune})

# 5. Run.
directory = 'images2'
process_images(directory)
