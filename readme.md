This project focuses on generating synthetic data and training an ai agent to judge the data.
generate_qa.py generates question/answer pairs about home repair. Run '.venv/bin/python generate_qa.py --count 50' to create 50 qa pairs--default is 20. Add '--output filename.jsonl' to change where it saves the data.

To run the judge, run '.venv/bin/python judge.py' to judge pairs. By default it judges all lines of qa_data.jsonl and prints to judge_results.jsonl. You can change this by adding --count number, --input filename.jsonl, and --output newfilename.jsonl.

To do a human review of the data, run '.venv/bin/python review_qa.py'. By default, it reads 'qa_data.jsonl' and outputs a file named 'human_labels.jsonl'.
