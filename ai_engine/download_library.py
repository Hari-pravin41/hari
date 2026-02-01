from web_scraper import WebLearner

def build_library():
    learner = WebLearner()
    
    # A curated list of classic literature with distinct, high-quality English
    library = [
        ("https://www.gutenberg.org/cache/epub/11/pg11.txt", "Alice in Wonderland"), # Imaginative
        ("https://www.gutenberg.org/cache/epub/84/pg84.txt", "Frankenstein"),        # Emotional/Deep
        ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "Pride and Prejudice"), # Formal/Social
        ("https://www.gutenberg.org/cache/epub/64317/pg64317.txt", "The Great Gatsby"), # Descriptive
    ]

    print("="*50)
    print(" BUILDING THE 'UNIVERSAL MIND' DATASET")
    print("="*50)

    for url, name in library:
        print(f"\n[Library] Fetching '{name}'...")
        learner.learn_from_url(url, source_name=name)

if __name__ == "__main__":
    build_library()
