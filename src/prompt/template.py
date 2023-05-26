# Define your prompt templates
prompt_templates = {
    'databricks/dolly-v2-12b': 
    """
    {context}\n
    {history}\n
    AI:
    """
    'checkpoint2': 'Template for checkpoint 2',
    'checkpoint3': 'Template for checkpoint 3'
}

# Function to generate prompts based on checkpoint key
def generate_prompt(checkpoint_key):
    template = prompt_templates.get(checkpoint_key)
    
    if template is None:
        # Handle invalid checkpoint key
        return "Invalid checkpoint key."
    
    # Process the template and generate the prompt
    prompt = process_template(template)
    
    return prompt

# Function to process the template (placeholder, add your own logic here)
def process_template(template):
    # Example placeholder logic: appending a string
    processed_template = template + ' Processed.'
    return processed_template

# Example usage
checkpoint_key = 'checkpoint2'
prompt = generate_prompt(checkpoint_key)
print(prompt)
