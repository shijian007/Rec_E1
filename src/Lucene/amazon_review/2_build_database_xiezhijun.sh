INPUT_DIR=/opt/data2/xiezhijun/code/Rec-R1/database/amazon_review/All_Beauty/jsonl_docs
INDEX_DIR=/opt/data2/xiezhijun/code/Rec-R1/database/amazon_review/All_Beauty/pyserini_index

python -m pyserini.index.lucene -collection JsonCollection \
 -input $INPUT_DIR \
 -index $INDEX_DIR \
 -generator DefaultLuceneDocumentGenerator \
 -threads 4 \
 -storePositions -storeDocvectors -storeRaw
