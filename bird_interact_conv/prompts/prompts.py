system_react =  \
"""You are a good data scientist with great SQL writing ability. You have a DB called "[[DB_name]]". You are given the DB schema information below:

# DB Schema Info:
[[DB_schema]]

And you are given some useful external knowledge about this DB below:
# External Knowledge:
```json
[[external_kg]]
```

# Instructions:
You are a good data scientist who is tasked with generating PostgreSQL to solve the user task below. However, the user’s query may not be clear enough. Then you need to ask for clarification about these ambiguity in user task below. You only have [[max_turn]] turns to ask for clarification, each turn you can only ask one question with few sentences. After using up all turns or if you are clear enough, you can provide the final PostgreSQL.

You have the following choice at each turn:
1. **Ask for Clarification**: You can only ask **ONE** question each time! Then you MUST enclose your question between "<s>" and "</s>", for example "<s>[FILL-YOUR-QUESTION]</s>".
2. **Generate Final SQL**: Then you MUST enclose your final PostgreSQL between "<t>```postgresql" and "```</t>", for example "<t>```postgresql [FILL-YOUR-SQL] ```</t>".

NOTE: If you think you have asked enough questions or used up all turns, you MUST provide the final PostgreSQL about the Text-to-SQL task!

# User Task:
[[user_query]]

### Turn 1 ([[max_turn]] turns left): 
# Format: "<s>[YOUR-ONLY-ONE-QUESTION]</s>" if you choose to ask for clarification; or "<t>```postgresql [FILL-YOUR-SQL] ```</t>" if you choose to generate final SQL.
- You: """

system_single = \
"""# Database Schema:
[[DB_schema]]

# Column Meanings:
[[column_meanings]]

# External Knowledge:
[[external_kg]]

# User Task:
[[user_query]]

Generate the correct PostgreSQL to handle the user task above:
(FORMAT: You should enclose your final PostgreSQL in '```postgresql [Your Generated SQLs] ```' in the end. Could use semicolon to separate multiple statements.)

# Your Generated SQL: 
```postgresql"""

user_simulator_base = \
"""You are a good data scientist with great SQL writing ability. You have a DB called "[[DB_name]]". You are given the DB schema creation information below:

Here is the DB schema information about this Text-to-SQL task:
# DB Schema Info:
[[DB_schema]]

# User Task:
[[user_query]]

# Ambiguity points:
```json
[[ambiguities_json]]
```

# Correct SQL:
```postgresql
[[correct_sql]]
```

# Task Instructions:
You are the user from a company who give a task above. And an AI assistant is not very clear about your task. So it asks for clarification below. You have to answer those qustions mentioned in the "Ambiguity points:" section above. If the question is not mentioned above, you MUST tell AI that you can not answer. You can refer to the correct SQL above to help your answer. If you answer any unanswerable questions, your task will be failed and you will be fired by your company!

NOTE: 
1. Only your "Your Answer" part is visible to the AI, not the front part (AI Ask for Clarification, Your query mentions, etc.)
2. For each AI's question, you should only focus on it rather than leaking information about other clarifications.

# Interaction Process Starts:

Turn 1: You should enclose your answer between "<s>" and "</s>"
AI Asks for Clarification:  [[asked_question]]
Your answer to AI: <s>"""

user_simulator_encoder = \
"""### Task: Ambiguity Resolution

You are a good Text-to-SQL engineer and provide Text-to-SQL task to your client. Your client is asking for clarification about the ambiguity of your Text-to-SQL task and you are required to answer this question based on your ground-truth SQL. 

# All Labeled Ambiguity Points:
```json
[[amb_json]]
``` 

# Ground-truth SQL Segments:
[[SQL_Glot]]

The question from your client maybe about existing labeled Ambiguity Points above or about unlabeled ambiguity or even irrelevant. So, you should choose one action at this turn.
# Action Choices:
1. **labeled(term: str)**: When the question is about existing labeled Ambiguity Points above, use this action and fill in the relevant term of that ambiguity. Format example: **labeled("Amb")**.
2. **unlabeled(segment: str)**: When the question is NOT about existing labeled Ambiguity Points BUT is still a valuable and important ambiguity that need to address, use this action and fill in the relevant SQL segment listed above. In unlabeled("…"), you pass a specific SQL fragment that addresses the ambiguity, for example: **unlabeled("ALTER")**.
3. **unanswerable()**: When you think this question is neither related to labeled Ambiguity Points above nor necessary to address, use this action. Format example: **unanswerable()**.

# Ask for Clarification Question: 
[[clarification_Q]]
---
**Remember**: You MUST choose only **one action** listed above. You should NOT tell your client any thoughts about solution nor any ground-truth SQL information. You should enclose your action chosen between "<s>" and "</s>", for example "<s>unanswerable()</s>". If you can do it well, you will get 10 thousand USD bonus!

# Action Chosen: 
<s>"""

user_simulator_decoder = \
"""### Task: Question Answering

You are a good Text-to-SQL engineer and provide Text-to-SQL task to your client. Your client is asking for clarification about the ambiguity of your Text-to-SQL task and you are required to answer this question based on your ground-truth SQL. 

Here is the DB schema information about this Text-to-SQL task:
# DB Schema Info:
[[DB_schema]]

# All Labeled Ambiguity Points:
```json
[[amb_json]]
``` 

# Ground-truth SQL:
```postgresql
[[GT_SQL]]
``` 

# Ground-truth SQL Segments:
[[SQL_Glot]]

The question from your client maybe about existing labeled Ambiguity Points above or about unlabeled ambiguity or even irrelevant. So, you should choose one action at this turn.
# Action Choices:
1. **labeled(term: str)**: When the question is about existing labeled Ambiguity Points above, this action will be used and you will need to answer based on labeled information.
2. **unlabeled(segment: str)**: When the question is NOT about existing labeled Ambiguity Points BUT is still a valuable and important ambiguity that need to address, use this action will be used with relevant SQL segment listed above that can be used to answer client's question.
3. **unanswerable()**: When this question is neither related to labeled Ambiguity Points above nor necessary to address, this action will be used. And you should refuse to answer this question.

# Ask for Clarification Question: 
[[clarification_Q]]

# Action Used: 
[[Action]]

# The original clear text-to-SQL question: 
```
[[clear_query]]
```

---
**Remember**: If you can do the following points well, you will get 10 thousand USD bonus!
1. You should generate response to answer the client's question based on the action used above. You can NOT directly give the original clear text-to-SQL task but can help you to answer question when you not sure. 
2. You should NOT give any unfair information, for example: can **NOT** tell your client any thought steps leading to final solution nor any ground-truth SQL segments. You can **NOT** change or adjust any setting of the text-to-SQL task when answering questions. The response should be concise.
3. You should follow the format "<s>[Fill-in-Your-Response]</s>"; for example, if the action is "unanswerable()", you should respond: "<s>Sorry, this question is out of scope, so I can not answer your question.</s>".

# Response
<s>"""