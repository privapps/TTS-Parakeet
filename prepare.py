import os, nltk

nltk.download('punkt')

with open('/workspace/__input__.txt', 'wt') as f:
    f.write(os.environ['INPUT_TEXT'])