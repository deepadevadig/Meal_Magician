from flask import Flask, render_template, request
import pandas as pd
import ast
import re
import os
import torch
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)  # ✅ Correct Flask initialization

# ✅ Load Pretrained BERT Model
bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load Dataset
try:
    df = pd.read_csv("data/cleaned_food.csv")
    print("✅ Dataset Loaded Successfully!")
except FileNotFoundError:
    print("❌ Dataset file not found! Using sample data for testing.")
    df = pd.DataFrame({
        'name': ['Sample Recipe'],
        'thumbnail_url': ['static/default_thumbnail.jpg'],
        'cleaned_ingredients': ['["potato", "onion", "garlic"]'],
        'instructions': ['[{"display_text": "Sample instruction step"}]'],
        'video_url': ['']
    })

# ✅ Ensure Correct Column Names
video_column = 'video_url' if 'video_url' in df.columns else 'youtube_url'
if video_column not in df.columns:
    df['video_url'] = ""

if 'thumbnail_url' not in df.columns:
    df['thumbnail_url'] = 'static/default_thumbnail.jpg'

# ✅ Create Static Folder (if it doesn't exist)
os.makedirs('static', exist_ok=True)

# ✅ Process Ingredients List
def process_ingredients(ingredients_str):
    """Convert ingredient strings into a list format."""
    if not isinstance(ingredients_str, str):
        return []
    try:
        ingredients = ast.literal_eval(ingredients_str)
        if isinstance(ingredients, list):
            return ingredients
    except (ValueError, SyntaxError):
        pass
    return ingredients_str.replace('[', '').replace(']', '').replace("'", '').split(', ')

df['cleaned_ingredients'] = df['cleaned_ingredients'].apply(process_ingredients)
df['ingredients_str'] = df['cleaned_ingredients'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')

# ✅ Compute BERT Embeddings for All Recipes (Precompute for Efficiency)
print("✅ Generating Recipe Embeddings...")
recipe_embeddings = bert_model.encode(df['ingredients_str'].tolist(), convert_to_tensor=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    
    if request.method == 'POST':
        user_input = request.form.get('ingredients', '').strip()
        if not user_input:
            return render_template('index.html', error="Please enter some ingredients", recommendations=[])

        user_input_list = [item.strip() for item in user_input.split(',') if item.strip()]
        if not user_input_list:
            return render_template('index.html', error="Please enter valid ingredients", recommendations=[])

        # ✅ Generate BERT Embedding for User Query
        user_input_str = ', '.join(user_input_list)
        user_embedding = bert_model.encode(user_input_str, convert_to_tensor=True)

        # ✅ Compute Similarity Between Query and Recipes
        cosine_similarities = util.pytorch_cos_sim(user_embedding, recipe_embeddings).squeeze(0)
        sorted_indices = torch.argsort(cosine_similarities, descending=True)
        
        # ✅ Select Top 5 Similar Recipes
        relevant_indices = [idx.item() for idx in sorted_indices if cosine_similarities[idx] > 0][:5]
        
        if not relevant_indices:
            return render_template('index.html', error="No matching recipes found", recommendations=[])

        recommendations = df.iloc[relevant_indices].to_dict(orient='records')

        # ✅ Process Additional Data
        for recipe in recommendations:
            recipe = process_recipe_instructions(recipe)
            recipe = process_video_url(recipe, video_column)

    return render_template('index.html', recommendations=recommendations)

def process_recipe_instructions(recipe):
    """Process the recipe instructions safely."""
    instructions_data = recipe.get('instructions', '')

    if isinstance(instructions_data, list) and all(isinstance(item, str) for item in instructions_data):
        recipe['instructions'] = instructions_data
        return recipe

    if isinstance(instructions_data, str):
        try:
            parsed_instructions = ast.literal_eval(instructions_data)
            if isinstance(parsed_instructions, list) and all(isinstance(item, dict) for item in parsed_instructions):
                recipe['instructions'] = [step.get('display_text', 'Step missing') for step in parsed_instructions]
                return recipe
        except (ValueError, SyntaxError):
            pass

    if isinstance(instructions_data, list) and all(isinstance(item, dict) for item in instructions_data):
        recipe['instructions'] = [step.get('display_text', 'Step missing') for step in instructions_data]
        return recipe

    recipe['instructions'] = ["Instructions not available"]
    return recipe

def process_video_url(recipe, video_column):
    """Processes video URLs for embedding."""
    url = recipe.get(video_column, '')

    if not isinstance(url, str) or not url.strip():
        recipe['embed_url'] = None
        recipe['video_type'] = None
        return recipe

    if "youtube.com/watch" in url:
        video_id = re.search(r'v=([^&]+)', url)
        if video_id:
            recipe['embed_url'] = f"https://www.youtube.com/embed/{video_id.group(1)}"
            recipe['video_type'] = "youtube"
            return recipe

    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0]
        recipe['embed_url'] = f"https://www.youtube.com/embed/{video_id}"
        recipe['video_type'] = "youtube"
        return recipe

    if url.endswith(".mp4"):
        recipe['embed_url'] = url
        recipe['video_type'] = "mp4"
        return recipe

    if "tasty.co" in url:
        recipe['embed_url'] = url
        recipe['video_type'] = "tasty"
        return recipe

    recipe['embed_url'] = None
    recipe['video_type'] = None
    return recipe

if __name__ == '__main__':
    app.run(debug=True)
