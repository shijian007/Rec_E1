INPUT_DIR=/opt/data2/xiezhijun/code/Rec-R1/database/rl_rec/mind_small_153k_v1_0423/jsonl_docs
INDEX_DIR=/opt/data2/xiezhijun/code/Rec-R1/database/rl_rec/mind_small_153k_v1_0423/pyserini_index

python -m pyserini.index.lucene -collection JsonCollection \
 -input $INPUT_DIR \
 -index $INDEX_DIR \
 -language zh \
 -generator DefaultLuceneDocumentGenerator \
 -threads 4 \
 -storePositions -storeDocvectors -storeRaw
