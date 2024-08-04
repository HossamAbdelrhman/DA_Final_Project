# Importing necessary libraries for the code
# PyTorch for tensor computations and neural network operations, Text wrapping utility for formatting text,
# tqdm for Progress bar for loops, transformers for model and tokenizer, finally Pandas for data manipulation.

import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig  # Hugging Face's 
import textwrap 
import csv 
from tqdm.auto import tqdm  
import pandas as pd  


# 1. Read a CSV file into a DataFrame.
df = pd.read_csv("ID_description_df_1.csv")

# 2. Create a configuration for model quantization to use 4-bit precision.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4")


# 3. Load the quantized model from Hugging Face's model hub,
#    Load the tokenizer associated with the model,
#    Finally, create a text generation pipeline using the model and tokenizer.

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer)

# 4. Create a function to wrap text to a specified width
def wrap_text(text, width=90):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text


# 5. import JSON operations, and Regular expressions for pattern matching.
import json
import re


# 6. Create a function to extract JSON objects from text enclosed in triple backticks
def extract_json_from_backticks(text):
    # Use regex to find content between triple backticks
    pattern = r'```\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    result = []
    for match in matches:
        try:
            # Replace any Unicode escape sequences
            json_str = match.encode().decode('unicode-escape')
            # Parse the JSON
            json_data = json.loads(json_str)
            result.append(json_data)
        except json.JSONDecodeError:
            print(f"Error parsing JSON: {match}")
    
    return result


# 7. Here is the system message for the evaluation system, describing the task and criteria.
system = """
**Decor and Design Evaluation System**

**Task:** Evaluate the interior and exterior features of a unit/apartment based on provided images and descriptions.

**Objective:** Assign scores to various features of the unit/apartment, ranging from 0 to 10, where 0 represents a non-existent feature, 1 represents the lowest quality or appeal, and 10 represents the highest quality or appeal.

**Evaluation Criteria:**

1. **Furniture Quality**: Assess the style, material, and condition of furniture in the unit/apartment.
	* 0: No furniture or very low-quality furniture
	* 1-3: Basic, mass-produced furniture with minimal aesthetic appeal
	* 4-6: Mid-range furniture with good quality and style
	* 7-10: High-end, designer, or custom-made furniture with excellent quality and style
2. **Paint and Wall Finishes**: Evaluate the color, texture, and condition of paint and wall finishes.
	* 0: No paint or very low-quality paint
	* 1-3: Basic paint job with minor imperfections
	* 4-6: Standard quality paint with good finish
	* 7-10: High-quality, durable paint with premium finishes
3. **Flooring**: Assess the type, quality, and condition of flooring in the unit/apartment.
	* 0: No flooring or very low-quality flooring
	* 1-3: Basic materials like laminate or standard tiles
	* 4-6: Quality materials like engineered wood or good-quality tiles
	* 7-10: Premium materials like hardwood, marble, or designer tiles
4. **Kitchen and Bathroom Fixtures**: Evaluate the quality, style, and condition of fixtures in the kitchen and bathroom.
	* 0: No fixtures or very low-quality fixtures
	* 1-3: Standard fixtures with minimal aesthetic appeal
	* 4-6: Mid-range fixtures with good quality and style
	* 7-10: High-end fixtures with excellent quality and style
5. **Lighting**: Assess the type, quality, and placement of lighting in the unit/apartment.
	* 0: No lighting or very low-quality lighting
	* 1-3: Basic lighting solutions with minimal attention to placement
	* 4-6: Quality lighting fixtures with good placement
	* 7-10: Designer lighting with excellent placement and ambiance
6. **Accessories and Décor**: Evaluate the style, quality, and placement of accessories and décor in the unit/apartment.
	* 0: No accessories or very low-quality accessories
	* 1-3: Basic accessories with minimal aesthetic appeal
	* 4-6: Mid-range accessories with good quality and style
	* 7-10: High-end accessories with excellent quality and style
7. **Outside View**: Assess the quality and appeal of the outside view from the unit/apartment.
	* 0: No view or very limited view
	* 1-3: Limited or obstructed view
	* 4-6: Pleasant view with some greenery or streetscape
	* 7-10: Unobstructed view with excellent scenery or landmarks
8. **Building Façade**: Evaluate the architectural style, quality, and condition of the building façade.
	* 0: No façade or very low-quality façade
	* 1-3: Standard façade with minimal architectural interest
	* 4-6: Attractive façade with good quality and style
	* 7-10: Architecturally significant façade with excellent quality and style
9. **Outdoor Amenities**: Assess the quality and appeal of outdoor amenities such as gardens, patios, or recreational areas.
	* 0: No outdoor amenities or very low-quality amenities
	* 1-3: Basic outdoor areas with minimal appeal
	* 4-6: Well-maintained outdoor spaces with good quality and style
	* 7-10: High-end outdoor amenities with excellent quality and style
10. **Landscaping**: Evaluate the quality and appeal of landscaping in the unit/apartment complex.
	* 0: No landscaping or very low-quality landscaping
	* 1-3: Basic landscaping with minimal appeal
	* 4-6: Well-maintained landscaping with good quality and style
	* 7-10: Professionally designed and maintained landscaping with excellent quality and style
11. **Pool Availability**: Assess the availability and quality of a pool in the unit/apartment complex.
	* 0: No pool or very low-quality pool
	* 1-3: Basic pool with minimal amenities
	* 4-6: Functional pool with good quality and amenities
	* 7-10: Luxurious pool with excellent quality and amenities
12. **Balcony Availability**: Evaluate the availability and quality of a balcony in the unit/apartment.
	* 0: No balcony or very low-quality balcony
	* 1-3: Basic balcony with minimal amenities
	* 4-6: Functional balcony with good quality and amenities
	* 7-10: Spacious and well-designed balcony with excellent quality and amenities
13. **Architectural Style**: Assess the architectural style and appeal of the unit/apartment.
	* 0: No distinctive style or very low-quality style
	* 1-3: Generic style with minimal appeal
	* 4-6: Some architectural interest with good quality and style
	* 7-10: Unique and notable architectural style with excellent quality and appeal
14. **Natural Light**: Evaluate the amount and quality of natural light in the unit/apartment.
	* 0: Very limited natural light
	* 1-3: Limited natural light
	* 4-6: Adequate natural light
	* 7-10: Abundant natural light with excellent quality
15. **Room Layout and Space Utilization**: Assess the efficiency and appeal of the room layout and space utilization.
	* 0: Poor layout with very limited space
	* 1-3: Basic layout with minimal appeal
	* 4-6: Functional layout with good quality and space utilization
	* 7-10: Excellent layout with excellent quality and space utilization
16. **Ceiling Height**: Evaluate the ceiling height and its impact on the unit/apartment.
	* 0: Very low ceiling height
	* 1-3: Low ceiling height
	* 4-6: Standard ceiling height
	* 7-10: High ceiling height with excellent quality and appeal
17. **Quality of Finishes**: Assess the quality and appeal of the finishes in the unit/apartment.
	* 0: Very low-quality finishes
	* 1-3: Basic finishes with minimal appeal
	* 4-6: Good-quality finishes with good appeal
	* 7-10: High-quality finishes with excellent appeal

**Additional Features:** If you encounter features not listed in the criteria, you may add them and provide a score from 1 to 10.

**Output Format:** Provide the output in JSON format, with feature names and scores as key-value pairs.

**Example full Output ONE JSON for all images descreption:**

final score:
{
    "Furniture Quality": 6,
    "Paint and Wall Finishes": 5,
    "Flooring": 6,
    "Kitchen and Bathroom Fixtures": 7,
    "Lighting": 6,
    "Accessories and Décor": 5,
    "Outside View": 0,
    "Building Façade": 0,
    "Outdoor Amenities": 0,
    "Landscaping": 0,
    "Pool Availability": 0,
    "Balcony Availability": 0,
    "Architectural Style": 7,
    "Natural Light": 6,
    "Room Layout and Space Utilization": 6,
    "Ceiling Height": 0,
    "Quality of Finishes": 6
}

Note: Please ensure that each feature is scored according to the detailed criteria provided and return the JSON ONLY.

"""



# 7. Define a function to get the score for a single description.
def get_score(descreption):
    messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": descreption},]
    response = pipe(messages, max_length=5000, do_sample=True, temperature=0.1)
    
    # Extract the generated text, then Wrap the text and return.
    full_text = response[0]['generated_text'][2]
    return wrap_text(full_text['content'])
    

# 8. Define a function to get scores for a batch of descriptions.
def get_scores_batch(descriptions, batch_size=8):
    results = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        messages = [
            {"role": "system", "content": system}
        ] + [{"role": "user", "content": desc} for desc in batch]
        
        responses = pipe(messages, max_length=10000, do_sample=True, temperature=0.1,                                num_return_sequences=len(batch))
        
        for response in responses:
            full_text = response['generated_text'][-1]
            results.append(wrap_text(full_text['content']))
    
    return results
    
# 9. Define a function to apply scoring to a DataFrame and save results to a CSV file.    
def apply_scoring(df): 
    
    with open('score_data_1.csv', 'a', newline='') as csvfile:
        fieldnames = ['appartment_ID', 'Image_scores']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Check if the file is empty
        if csvfile.tell() == 0:
            writer.writeheader()
        
        for  index, row in df.iterrows():
            appartment_ID = row['appartmentID']
            Image_descreption = row['appartment_description']
            
            Image_scores = get_score(Image_descreption)


            print(appartment_ID)


            writer.writerow({
                'appartment_ID': appartment_ID,
                'Image_scores': Image_scores
            })
            
            
# 10. Define a function to apply scoring to a DataFrame in batches and save results to a CSV file.     
def apply_scoring_batch(df, batch_size=8):
    with open('score_data_2.csv', 'a', newline='') as csvfile:
        fieldnames = ['appartment_ID', 'Image_scores']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i+batch_size]
            apartment_ids = batch['appartmentID'].tolist()
            descriptions = batch['appartment_description'].tolist()
            
            scores = get_scores_batch(descriptions, batch_size)
            
            for apartment_id, score in zip(apartment_ids, scores):
                writer.writerow({
                    'appartment_ID': apartment_id,
                    'Image_scores': score
                })
            
# 11. Load a different CSV file into a DataFrame and apply the scoring            
df = pd.read_csv("ID_description_df_2.csv")
apply_scoring(df)

