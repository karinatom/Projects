"""Configuration and parameters."""

BATCH_SIZE = 512
MAX_EPOCHS = 50

MAX_NUM_WORDS = 20000
SONGDATA_FILE = "./data/songdata.csv"
NUM_LINES_TO_INCLUDE = 4
MAX_REPEATS = 2
SAVE_FREQUENCY = 10
EARLY_STOPPING_PATIENCE = 5

# The default embedding dimension matches the glove filename
EMBEDDING_DIM = 50
EMBEDDING_FILE = "./data/glove.6B.50d.txt"

# Sample rock artists (this was based on a random top 20 I found online)
# Artists are confirmed to exist in the dataset
ARTISTS = [
     'alice_in_chains',
      'alice_in_chains-rooster',
      'audioslave',
      'nine_inch_nails',
      'pearl_jam',
      'perfect_circle',
      'soundgarden',
      'tool'
    ]
