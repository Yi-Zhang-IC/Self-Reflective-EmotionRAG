import os
import requests
from bs4 import BeautifulSoup
import json
from time import sleep

# === Editable list of characters ===
character_names = [
    "Severus Snape",
    "Harry Potter",
    "Albus Dumbledore",
    "Luna Lovegood",
    "Hermione Granger",
    "Ron Weasley",
    "Draco Malfoy",
    "Minerva McGonagall"
]

def extract_biography_for_character(character_name):
    # Convert name to Fandom URL format and lowercase file-safe name
    fandom_name = character_name.replace(" ", "_")
    folder_name = fandom_name.lower()

    # Target URL
    url = f"https://harrypotter.fandom.com/wiki/{fandom_name}"
    print(f"🔍 Fetching: {character_name} → {url}")

    # Setup folder structure
    base_dir = f"database/{folder_name}"
    os.makedirs(base_dir, exist_ok=True)

    raw_paragraphs_path = os.path.join(base_dir, "raw_paragraphs.json")
    memory_output_path = os.path.join(base_dir, "memory.json")

    # Fetch HTML content
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find "Biography" section
    bio_heading = soup.find(lambda tag: tag.name in ['h2', 'h3'] and 'Biography' in tag.text)
    if not bio_heading:
        print(f"Biography section not found for {character_name}")
        return

    # Extract ordered paragraphs from Biography section
    paragraphs = []
    current = bio_heading.find_next_sibling()
    while current and current.name != 'h2':
        if current.name == 'p':
            text = current.get_text().strip()
            if text:
                paragraphs.append(text)
        current = current.find_next_sibling()

    # Sort by appearance order and build structured DB
    memory_db = [
        {"paragraph_index": i, "raw_text": para}
        for i, para in enumerate(paragraphs)
    ]

    # Save raw paragraphs
    with open(raw_paragraphs_path, "w", encoding="utf-8") as f:
        json.dump(memory_db, f, indent=2, ensure_ascii=False)

    # Create empty memory.json (to be filled later)
    if not os.path.exists(memory_output_path):
        with open(memory_output_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    print(f"Saved {len(paragraphs)} paragraphs for '{character_name}'")

# === Loop through all characters ===
for name in character_names:
    try:
        extract_biography_for_character(name)
        sleep(2)  # polite delay to avoid hammering Fandom servers
    except Exception as e:
        print(f"Error processing {name}: {e}")
