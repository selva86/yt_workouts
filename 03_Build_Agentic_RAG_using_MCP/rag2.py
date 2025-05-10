# Building an Intelligent Q&A System with Vector Search

"""
Hey there! Let's build a smart Q&A system together. Think of it like creating
your own personal assistant that can understand questions and find the best
answers from a knowledge base. We'll use some cool tech to make this happen!

First, we'll take some FAQ data, turn it into numbers (embeddings), store it
in a database, and then search through it to find the best answers to questions.
"""

# Standard imports - these are our toolbox!
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
from qdrant_client import models
from qdrant_client import QdrantClient

# -----------------------------------------------------------------------------
# STEP 1: Prepare our data
# -----------------------------------------------------------------------------

def load_faq_data():
    """
    This function loads our FAQ data. Think of it as our knowledge base.
    We have 20 questions and answers on F1 Racing that our system will search through.
    """
    faq_content = """Question 1: What does F1 stand for in Formula 1 racing?
    Answer 1: Formula One refers to the specific set of rules and regulations that all participating cars must follow, making it the highest class of single-seater auto racing.

    Question 2: How many teams compete in a typical F1 season?
    Answer 2: Usually 10 teams, with each team fielding two cars, making a total of 20 drivers on the grid.

    Question 3: What is the significance of tire compounds in F1?
    Answer 3: Different tire compounds offer varying levels of grip and durability, with softer compounds providing more grip but wearing out faster.

    Question 4: How long is a standard F1 race distance?
    Answer 4: Races are typically around 305 kilometers (190 miles), except Monaco which is shorter at 260 kilometers.

    Question 5: What is DRS and when can drivers use it?
    Answer 5: Drag Reduction System - a moveable rear wing element that reduces drag. Drivers can activate it in designated zones when within one second of the car ahead.

    Question 6: What determines the starting grid positions for a race?
    Answer 6: Qualifying sessions where drivers compete in three knockout rounds (Q1, Q2, Q3) to determine their starting positions.
    
    Question 7: How many points are awarded for winning an F1 race?
    Answer 7: 25 points for the winner, with decreasing points down to 1 point for 10th place, plus an additional point for the fastest lap.

    Question 8: What is the purpose of the pit stop in F1?
    Answer 8: To change tires, refuel (in past eras), make repairs, or adjust the car's setup during the race.

    Question 9: What is understeer and oversteer in F1 driving?
    Answer 9: Understeer occurs when the front wheels lose grip and the car doesn't turn enough; oversteer happens when the rear loses grip and the car turns too much.

    Question 10: How do F1 cars generate downforce?
    Answer 10: Through aerodynamic elements like wings, diffusers, and the car's floor design that create low pressure under the car, pulling it toward the track.

    Question 11: What is the role of the F1 Safety Car?
    Answer 11: To slow down the field during dangerous conditions, allowing marshals to clear incidents safely while maintaining the race order.

    Question 12: How much do F1 cars weigh?Answer 12: The minimum weight including the driver is 798 kg (1,759 lbs), though teams often add ballast to reach this minimum for better weight distribution.

    Question 13: What is parc fermé in F1?
    Answer 13: A period from qualifying to the race start where teams cannot make significant setup changes to their cars.

    Question 14: How fast can F1 cars accelerate?
    Answer 14: They can accelerate from 0 to 100 km/h (62 mph) in approximately 2.4 seconds.

    Question 15: What is the difference between wet and intermediate tires?
    Answer 15: Intermediate tires are for damp conditions with standing water, while full wet tires are designed for heavy rain with deeper treads.

    Question 16: Who governs F1 racing?
    Answer 16: The FIA (Fédération Internationale de l'Automobile) sets the rules and regulations for Formula 1.

    Question 17: What is the halo device on F1 cars?
    Answer 17: A titanium safety structure above the cockpit designed to protect drivers' heads from debris and impacts.

    Question 18: How are F1 engines different from road car engines?
    Answer 18: F1 engines are highly sophisticated hybrid power units with turbocharging, energy recovery systems, and can rev up to 15,000 RPM.

    Question 19: What is slipstreaming in F1?
    Answer 19: Following closely behind another car to benefit from reduced air resistance, allowing for higher speeds and potential overtaking opportunities.

    Question 20: How does the championship points system work?
    Answer 20: Drivers and constructors accumulate points throughout the season based on race finishes, with the highest point totals winning their respective championships.
    """
    return faq_content

def clean_and_split_faq(faq_text):
    """
    This function takes our raw FAQ text and converts it into a clean list.
    Each item in the list is one Q&A pair that we can search through later.
    We replace any line breaks with spaces to keep things tidy.
    """
    # Split the text by double line breaks (which separate Q&A pairs)
    qa_pairs = faq_text.split("\n\n")
    
    # Clean up each pair by replacing single line breaks with spaces
    cleaned_pairs = [pair.replace("\n", " ") for pair in qa_pairs]
    
    return cleaned_pairs

# -----------------------------------------------------------------------------
# STEP 2: Create helper functions for batch processing
# -----------------------------------------------------------------------------

def create_batches(data_list, batch_size):
    """
    This is a handy utility function that splits a big list into smaller chunks.
    Think of it like dividing a big pizza into slices - easier to handle!
    We use this when processing data in batches to avoid overwhelming our system.
    """
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]

# -----------------------------------------------------------------------------
# STEP 3: Create our embedding system
# -----------------------------------------------------------------------------

class EmbeddingGenerator:
    """
    This class is responsible for converting text into numbers (vectors).
    It's like translating text into a language that computers can understand.
    We use these numbers to find similar content later on.
    """
    
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5", batch_size=32):
        """
        Initialize our embedding generator with a specific model.
        The model we're using is good at understanding text meaning.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = self._initialize_model()
        self.embeddings = []
        self.text_data = []
    
    def _initialize_model(self):
        """
        Load the embedding model from HuggingFace.
        We store it locally to avoid downloading it every time.
        """
        return HuggingFaceEmbedding(
            model_name=self.model_name,
            trust_remote_code=True,
            cache_folder='./hf_cache'
        )
    
    def convert_text_to_vectors(self, text_list):
        """
        This is where the magic happens! We convert text to vectors.
        Each piece of text becomes a list of 768 numbers that represent its meaning.
        """
        return self.model.get_text_embedding_batch(text_list)
    
    def process_all_data(self, text_list):
        """
        Process all our text data in batches.
        We do this in batches to be memory-efficient and show progress.
        """
        self.text_data = text_list
        
        # Calculate how many batches we'll need
        total_batches = len(text_list) // self.batch_size + (1 if len(text_list) % self.batch_size > 0 else 0)
        
        # Process each batch
        for batch in tqdm(create_batches(text_list, self.batch_size), 
                         total=total_batches,
                         desc="Converting text to vectors"):
            
            batch_vectors = self.convert_text_to_vectors(batch)
            self.embeddings.extend(batch_vectors)
        
        print(f"✅ Successfully converted {len(self.embeddings)} items to vectors!")

# -----------------------------------------------------------------------------
# STEP 4: Create our database system
# -----------------------------------------------------------------------------

class VectorDatabase:
    """
    This class manages our vector database using Qdrant.
    Think of it as a smart filing cabinet that can quickly find similar documents.
    """
    
    def __init__(self, collection_name, vector_size=768, batch_size=512):
        """
        Initialize our database connection and settings.
        """
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_size = vector_size
        self.client = None
        self._connect_to_database()
    
    def _connect_to_database(self):
        """
        Connect to our local Qdrant database.
        Qdrant must be running on your machine for this to work.
        """
        self.client = QdrantClient(
            url="http://localhost:6333",
            prefer_grpc=True
        )
        print("✅ Connected to vector database!")
    
    def _collection_exists(self):
        """Check if our collection already exists."""
        return self.client.collection_exists(collection_name=self.collection_name)
    
    def create_storage_space(self):
        """
        Create a new collection if it doesn't exist.
        This is like creating a new folder in our filing cabinet.
        """
        if not self._collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.DOT,  # How we measure similarity
                    on_disk=True  # Store on disk to save memory
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=0  # Start indexing immediately
                )
            )
            print(f"✅ Created new collection: {self.collection_name}")
        else:
            print(f"ℹ️  Collection '{self.collection_name}' already exists")
    
    def store_embeddings(self, embedding_generator):
        """
        Save all our embeddings and their corresponding text to the database.
        We do this in batches to be efficient.
        """
        # Calculate total batches for progress tracking
        total_items = len(embedding_generator.text_data)
        total_batches = total_items // self.batch_size + (1 if total_items % self.batch_size > 0 else 0)
        
        # Process in batches
        for text_batch, vector_batch in tqdm(
            zip(create_batches(embedding_generator.text_data, self.batch_size),
                create_batches(embedding_generator.embeddings, self.batch_size)),
            total=total_batches,
            desc="Storing data in database"
        ):
            
            # Upload this batch to Qdrant
            self.client.upload_collection(
                collection_name=self.collection_name,
                vectors=vector_batch,
                payload=[{"content": text} for text in text_batch]
            )
        
        # Optimize the collection for faster searches
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
        )
        print(f"✅ Successfully stored {total_items} items in the database!")

# -----------------------------------------------------------------------------
# STEP 5: Create our search system
# -----------------------------------------------------------------------------

class SmartSearcher:
    """
    This class handles searching through our stored data.
    It's like having a very fast librarian who can find exactly what you're looking for.
    """
    def __init__(self, database, embedding_generator):
        """
        Initialize with our database and embedding system.
        """
        self.database = database
        self.embedding_generator = embedding_generator
    
    def find_similar_content(self, user_question, top_k=3):
        """
        Find the most similar content to a user's question.
        
        Steps:
        1. Convert the question to a vector
        2. Search the database for similar vectors
        3. Return the original text of the most similar items
        """
        # Convert the user's question to a vector
        question_vector = self.embedding_generator.model.get_query_embedding(user_question)
        
        # Search the database
        search_results = self.database.client.search(
            collection_name=self.database.collection_name,
            query_vector=question_vector,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=True,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            limit=top_k,
            timeout=1000
        )
        
        # Extract the text content from search results
        relevant_content = []
        for result in search_results:
            content = result.payload.get("content", "")
            if content:
                relevant_content.append(content)
        
        # Combine all relevant content with separators
        combined_answer = "\n\n---\n\n".join(relevant_content[:top_k])
        return combined_answer

# -----------------------------------------------------------------------------
# STEP 6: Put it all together!
# -----------------------------------------------------------------------------

def build_qa_system():
    """
    This is our main function that builds the entire Q&A system.
    It ties all the pieces together in the right order.
    """
    print("🚀 Building your intelligent Q&A system...\n")
    
    # Step 1: Load and prepare the data
    print("📚 Loading FAQ data...")
    faq_text = load_faq_data()
    qa_pairs = clean_and_split_faq(faq_text)
    print(f"✅ Loaded {len(qa_pairs)} Q&A pairs\n")
    
    # Step 2: Create embeddings
    print("🔢 Converting text to vectors...")
    embedding_gen = EmbeddingGenerator(batch_size=32)
    embedding_gen.process_all_data(qa_pairs)
    print(f"✅ Created {len(embedding_gen.embeddings)} embeddings of dimension {len(embedding_gen.embeddings[0])}\n")
    
    # Step 3: Set up the database
    print("🗄️  Setting up vector database...")
    database = VectorDatabase("ml_faq_collection")
    database.create_storage_space()
    database.store_embeddings(embedding_gen)
    print()
    
    # Step 4: Create the searcher
    print("🔍 Setting up search system...")
    searcher = SmartSearcher(database, embedding_gen)
    print("✅ Search system ready!\n")
    
    print("🎉 Your Q&A system is ready to go!")
    return searcher

def ask_question(searcher, question):
    """
    A simple function to ask questions and get answers.
    This makes it easy to use our Q&A system.
    """
    print(f"\n❓ Question: {question}")
    print("-" * 50)
    
    answer = searcher.find_similar_content(question)
    
    print("📝 Answer:")
    print(answer)
    print("-" * 50)
    
    return answer

# -----------------------------------------------------------------------------
# Running the system
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Build the system
    searcher = build_qa_system()
    
    # Example usage
    test_questions = [
        "How to prevent overfitting?",
        "What should I do with missing values?",
        "When is deep learning useful?"
    ]
    
    print("\n" + "="*60)
    print("Testing our Q&A system with some questions...")
    print("="*60)
    
    for question in test_questions:
        ask_question(searcher, question)