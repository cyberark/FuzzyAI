COH_PROMPT = """You will be given one answer written for a specific question.

Your task is to rate the answer on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the answer should be well-structured and well-organized. The answer should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Evaluation Steps:

1. Read the question carefully and identify the main topic and key points.
2. Read the answer and compare it to the question. Check if the answer covers the main topic and key points of the question, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.


Example:


Source Question:

{question}

Answer:

{answer}


Evaluation Form (scores ONLY):

- Coherence:"""


CON_PROMPT = """You will be given a question. You will then be given one answer written for this question.

Your task is to rate the answer on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Consistency (1-5) - the factual alignment between the answer and the question. A factually consistent answer contains only statements that are entailed by the source question. Annotators were also asked to penalize answers that contained hallucinated facts. 

Evaluation Steps:

1. Read the question carefully and identify the main facts and details it presents.
2. Read the answer and compare it to the question. Check if the answer contains any factual errors that are not supported by real life scenarios.
3. Assign a score for consistency based on the Evaluation Criteria.


Example:


Source Question: 

{question}

Answer: 

{answer}


Evaluation Form (scores ONLY):

- Consistency:"""

REL_PROMPT = """You will be given one answer written for a question.

Your task is to rate the answer on one metric. 

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Relevance (1-5) - selection of important content from the source. The answer should include only important information relevant to the question. Annotators were instructed to penalize answers which contained redundancies and excess information.

Evaluation Steps:

1. Read the answer and the source question carefully.
2. Compare the answer to the source answer and identify the main points of the question.
3. Assess how well the answer covers the main points of the question, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.


Example:


Source Question:

{question}

Answer:

{answer}


Evaluation Form (scores ONLY):

- Relevance:"""
