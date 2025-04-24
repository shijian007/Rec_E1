# Building a Database with Lucene

## Step 1: Convert the original data format into the Lucene format
Run the code
```[python]
python code/src/Lucene/scifact/1_convert_format.py
```
You should modify the path in the file accordingly.

## Step 2: Build the database
Run
```[bash]
bash code/src/Lucene/scifact/2_build_database.sh
```

## Step 3: Test the database search
Refer to `code/src/Lucene/scifact/search.py` file.