#!/usr/bin/env python3
"""
Generate comprehensive summaries for items in Beauty.pretrain.json using GPT-4o-mini.
Uses LangChain's batch API for efficient processing.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def load_beauty_items(beauty_items_file: Path) -> dict:
    """Load the beauty items JSON file."""
    print(f"Loading Beauty items file: {beauty_items_file}")
    with beauty_items_file.open("r", encoding="utf-8") as f:
        beauty_items = json.load(f)
    print(f"Loaded {len(beauty_items)} items")
    return beauty_items


def create_summary_prompt() -> ChatPromptTemplate:
    """Create the prompt template for generating item summaries."""
    template = """You are a beauty and personal care product expert. Generate a comprehensive, informative summary (approximately 1000 words) for the following product.

Product Information:
- Title: {title}
- Description: {description}
- Categories: {categories}

Your summary should include:

1. **Product Overview** (150-200 words)
   - What the product is and its primary purpose
   - Key features and benefits mentioned in the description
   
2. **Key Ingredients & Formulation** (200-250 words)
   - Based on the product name and description, identify and explain key ingredients
   - How these ingredients work and their benefits
   - Any notable formulation characteristics (e.g., sulfate-free, organic, etc.)
   
3. **Usage & Application** (150-200 words)
   - How to use the product effectively
   - When to use it (daily routine, specific occasions, etc.)
   - Tips for best results
   
4. **Suitable For** (100-150 words)
   - Target audience (skin type, hair type, age group, etc.)
   - Who would benefit most from this product
   - Any considerations or warnings
   
5. **Product Benefits & Results** (150-200 words)
   - Expected outcomes from regular use
   - Short-term vs long-term benefits
   - How it addresses specific concerns
   
6. **Common User Feedback & Reception** (150-200 words)
   - Based on your knowledge of similar products, what do users typically appreciate
   - Common concerns or considerations
   - Why this type of product is popular or effective
   
7. **Product Category Context** (100-150 words)
   - Where this product fits in the {categories} category
   - How it compares to similar products in its category
   - Industry trends related to this product type

Write the summary in a professional, informative tone. Be specific and detailed, drawing on both the provided information and your knowledge of beauty and personal care products.

Generate the comprehensive summary now:"""

    return ChatPromptTemplate.from_template(template)


def prepare_batch_inputs(beauty_items: dict, max_items: int = None) -> List[Dict[str, Any]]:
    """Prepare inputs for batch processing."""
    inputs = []
    
    items_to_process = list(beauty_items.items())
    if max_items:
        items_to_process = items_to_process[:max_items]
    
    for item_id, item_info in items_to_process:
        title = item_info.get("title", "Unknown Product")
        description = item_info.get("description", "No description available")
        categories = item_info.get("categories", "Unknown Category")
        
        inputs.append({
            "item_id": item_id,
            "title": title,
            "description": description,
            "categories": categories,
        })
    
    return inputs


def generate_summaries_batch(
    inputs: List[Dict[str, Any]],
    api_key: str,
    batch_size: int = 10,
    model: str = "gpt-4o-mini",
    output_file: Path = None,
    beauty_items: dict = None,
) -> Dict[str, str]:
    """Generate summaries using batch API."""
    
    print(f"Initializing ChatOpenAI with model: {model}")
    
    # Initialize ChatOpenAI
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.7,
        # No max_tokens constraint - let the model decide based on prompt
    )
    
    # Create prompt and chain
    prompt = create_summary_prompt()
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    summaries = {}
    
    # Process in batches
    print(f"Processing {len(inputs)} items in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(inputs), batch_size), desc="Processing batches"):
        batch = inputs[i:i + batch_size]
        
        # Prepare batch inputs for the chain
        batch_chain_inputs = [
            {
                "title": item["title"],
                "description": item["description"],
                "categories": item["categories"],
            }
            for item in batch
        ]
        
        try:
            # Use batch API with max_concurrency for parallel processing
            batch_results = chain.batch(
                batch_chain_inputs,
                config={"max_concurrency": 50}  # Process up to 50 items concurrently
            )
            
            # Store results
            for item, summary in zip(batch, batch_results):
                summaries[item["item_id"]] = summary
            
            # Save incrementally every 10 batches (500 items)
            if output_file and beauty_items and ((i // batch_size) % 10 == 9 or i + batch_size >= len(inputs)):
                save_summaries(beauty_items, summaries, output_file)
                
        except Exception as e:
            print(f"\nError processing batch {i//batch_size}: {e}")
            print("Continuing with next batch...")
            continue
    
    print(f"\nSuccessfully generated {len(summaries)} summaries")
    return summaries


def save_summaries(
    beauty_items: dict,
    summaries: Dict[str, str],
    output_file: Path,
) -> None:
    """Save the beauty items with added summaries to a new JSON file."""
    
    print(f"Adding summaries to beauty items...")
    updated_items = {}
    
    for item_id, item_info in beauty_items.items():
        updated_item = item_info.copy()
        if item_id in summaries:
            updated_item["ai_summary"] = summaries[item_id]
        updated_items[item_id] = updated_item
    
    print(f"Saving to: {output_file}")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(updated_items, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(updated_items)} items with summaries")
    print(f"  - Items with AI summaries: {len(summaries)}")
    print(f"  - Items without summaries: {len(updated_items) - len(summaries)}")


def main():
    # Configuration
    BEAUTY_ITEMS_FILE = Path("./Beauty.pretrain.json")
    OUTPUT_FILE = Path("./Beauty.pretrain.with_summaries.json")
    API_KEY = os.environ.get("OAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not API_KEY:
        raise RuntimeError("Please set OAI_API_KEY or OPENAI_API_KEY.")
    MODEL = "gpt-4.1-mini"  # Using gpt-4.1-mini
    
    # Batch processing settings
    BATCH_SIZE = 200  # Process 200 items at a time
    MAX_ITEMS = None  # Set to None to process all items, or a number for testing
    
    print("=" * 80)
    print("Beauty Product Summary Generation")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max items: {MAX_ITEMS if MAX_ITEMS else 'All'}")
    print()
    
    # Load beauty items
    beauty_items = load_beauty_items(BEAUTY_ITEMS_FILE)
    
    # Prepare inputs
    print("\nPreparing batch inputs...")
    inputs = prepare_batch_inputs(beauty_items, max_items=MAX_ITEMS)
    print(f"Prepared {len(inputs)} items for processing")
    
    # Generate summaries
    print("\nGenerating summaries with GPT-4o-mini...")
    summaries = generate_summaries_batch(
        inputs=inputs,
        api_key=API_KEY,
        batch_size=BATCH_SIZE,
        model=MODEL,
        output_file=OUTPUT_FILE,
        beauty_items=beauty_items,
    )
    
    # Final save
    print("\nSaving final results...")
    save_summaries(beauty_items, summaries, OUTPUT_FILE)
    
    print("\n" + "=" * 80)
    print("✓ Summary generation completed!")
    print("=" * 80)
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print("\nYou can now use this file with the 'ai_summary' field for each item.")


if __name__ == "__main__":
    main()

