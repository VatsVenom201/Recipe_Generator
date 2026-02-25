# prompt_ui.py
import os
import warnings
warnings.filterwarnings("ignore")

# SHARED CACHE LOCATION (use this in ALL your scripts)
CACHE_DIR = r"D:\huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR

print(f"Using shared cache: {CACHE_DIR}")

# ----------------- 1️⃣ IMPORTS -----------------
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


# ----------------- 2️⃣ LOAD LOCAL MODEL (4GB VRAM Optimized) -----------------
@st.cache_resource  # Cache the model so it doesn't reload on every interaction
def load_local_llm():
    """Load SmolLM2-1.7B model locally with FP16"""

    print(f"Loading model from cache: {CACHE_DIR}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in FP16 (fits in 4GB VRAM)
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        dtype=torch.float16,
        device_map="cuda:0",
    )

    # Create pipeline with enough tokens for complete recipes
    transformers_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,  # Increased for complete recipes
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
    )

    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=transformers_pipeline)
    return llm


# ----------------- 3️⃣ PROMPT TEMPLATE -----------------
prompt = PromptTemplate(
    input_variables=["ingredient_list"],
    template="""<|im_start|>system
You are a professional chef. Create detailed, easy-to-follow recipes with cooking steps, estimated time, serving suggestions, and tips.
<|im_start|>user
Create a recipe using these ingredients:
{ingredient_list}
<|im_start|>assistant"""
)

# ----------------- 4️⃣ STREAMLIT UI -----------------
st.set_page_config(page_title="AI Recipe Generator", page_icon="🍳")
st.title("🍳 AI Recipe Generator")
st.caption("Running entirely on your GPU")

# Initialize session state
if "ingredients" not in st.session_state:
    st.session_state.ingredients = []
if "recipe_generated" not in st.session_state:
    st.session_state.recipe_generated = None


# Function to add ingredient
def add_ingredient():
    st.session_state.ingredients.append({"name": "", "quantity": "", "type": "Protein"})


# Add default ingredient if empty
if len(st.session_state.ingredients) == 0:
    add_ingredient()

st.subheader("Add Your Ingredients")

# Display ingredient inputs
for i, ingredient in enumerate(st.session_state.ingredients):
    cols = st.columns([3, 2, 2, 1])
    ingredient["name"] = cols[0].text_input(
        f"Name",
        value=ingredient["name"],
        key=f"name_{i}",
        placeholder="e.g., Chicken"
    )
    ingredient["quantity"] = cols[1].text_input(
        f"Quantity",
        value=ingredient["quantity"],
        key=f"quantity_{i}",
        placeholder="e.g., 200g"
    )
    ingredient["type"] = cols[2].selectbox(
        f"Type",
        options=["Protein", "Vegetable", "Fruit", "Spice", "Dairy", "Grain", "Other"],
        index=["Protein", "Vegetable", "Fruit", "Spice", "Dairy", "Grain", "Other"].index(ingredient["type"]),
        key=f"type_{i}"
    )
    # Delete button
    if cols[3].button("🗑️", key=f"del_{i}"):
        st.session_state.ingredients.pop(i)
        st.rerun()

# Add ingredient button
col1, col2 = st.columns([1, 3])
with col1:
    st.button("➕ Add Ingredient", on_click=add_ingredient)

with col2:
    # Clear all button
    if st.button("Clear All"):
        st.session_state.ingredients = []
        st.session_state.recipe_generated = None
        st.rerun()

# Generate recipe button
st.divider()
if st.button("🍳 Generate Recipe", type="primary", use_container_width=True):
    # Validate
    valid_ingredients = [ing for ing in st.session_state.ingredients
                         if ing["name"].strip() and ing["quantity"].strip()]

    if not valid_ingredients:
        st.error("Please enter at least one ingredient with name and quantity.")
    else:
        # Format ingredients
        ingredient_list_str = "\n".join([
            f"- {ing['name']}: {ing['quantity']} ({ing['type']})"
            for ing in valid_ingredients
        ])

        # Load model (cached) and generate
        with st.spinner("Chef is cooking... (Loading model first time may take 1-2 mins)"):
            try:
                llm = load_local_llm()
                llm_chain = LLMChain(prompt=prompt, llm=llm)
                recipe_output = llm_chain.invoke({"ingredient_list": ingredient_list_str})
                st.session_state.recipe_generated = recipe_output['text']
            except Exception as e:
                st.error(f"Error generating recipe: {e}")
                st.info("If this is your first run, the model is downloading. Check your D: drive space!")

# Display result
if st.session_state.recipe_generated:
    st.divider()
    st.subheader("🍽️ Your Recipe")
    st.markdown(st.session_state.recipe_generated)

    # Download button
    st.download_button(
        label="📥 Download Recipe",
        data=st.session_state.recipe_generated,
        file_name="recipe.txt",
        mime="text/plain"
    )