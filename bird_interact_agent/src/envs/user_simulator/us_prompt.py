# for chat model as user.
AGENT_INIT_MESSAGE = "Hi! How can I help you today?"

# for non-chat model
user_prompt_template = """Your task is to simulate a human user that interacts with an LLM assistant in a dialogue.
You would like the LLM assistant to help you with the a NL query to get the correct SQL query.

Your goal is to engage in the conversation with the LLM assistant so that it can get to a clarified SQL query.
You should make use of the following hidden information about annotated potential ambiguities and the key sql snippet as clarification to answer the LLM assistant's questions.
YOU SHOULD BEHAVE LIKE A END USER THAT NEEDS HELP FROM AN ASSISTANT, SO YOUR CLARIFICATION SHOULD BE CONCISE AND IN NATURAL LANGUAGE.
YOU SHOULD ONLY ANSWER QUESTIONS WITH INFORMATION PROVIDED IN THE HIDDEN INFORMATION AND THE REFERENCE SQL.
IF THE ANSWER CANNOT BE FOUND IN THE HIDDEN INFORMATION OR THE REFERENCE SQL, SAY YOU DON’T KNOW.
DON'T EXPOSE THE REFERENCE SQL TO THE LLM ASSISTANT.

Your Query: {ambiguous_query}

Hidden Information (only visible to you):
> Reference SQL (the actual SQL query that you want to get):
{reference_sql}

> Potential ambiguities terms in your query and key SQL snippets as clarification:
{clarifications}
 
------------------------------
Here is the dialogue between you and the LLM assistant so far:
{dialogue_history}

------------------------------
| Now you serve as user, directly outputting to answer to the LLM assitant IN TWO SENTENCES. DO NOT SAY ANYTHING ELSE.

User (You): """


# for chat model. Prompt is referenced from sweet_rl repo (https://github.com/facebookresearch/sweet_rl/blob/main/prompts/human_simulator_code_prompt.txt)
# Needed to be refined.
user_prompt_system_template = """
<Database Schema>
A database `{db_name}` has the following schema and some example data:
```sql
{db_schema}
```
</Database Schema>

----------------Instruction----------------
BACKGROUND: You are a DB end-user asking a LLM-based assistant for help to convert a natural language request into the correct SQL query. You have a clear intention, but your request expressed in natural language is ambiguous and the assistant is not very clear about your real intention. 

TASK: You are testing whether the assistant can resolve the ambiguity in your request through a dialogue and finally give the correct SQL query. Each round, the assistant may ask you some questions to clarify your intention, and you should answer it based on your real intention.

Your real intention is provided to you in the <Hidden Information>, containing the reference SQL query, and the potential ambiguity terms in your request with the key SQL snippets for your reference to clarify your intention.

-----------------Information-----------------
YOUR REQUEST: "{ambiguous_query}"

> External Knowledge involved in your request: (For assistant, they need to retrieve these knowledge from a KB document)
{golden_kb}

<Hidden Information> (Your real intention, only visible to you):
> Reference SQL (the actual SQL query that you want to get):
```sql
{reference_sql}
```

> Potential ambiguities in your request and key SQL snippets as reference to help your clarification:
{clarifications}
</Hidden Information>

----------------Note----------------
REMEMBER:
- You are the USER (not the system, not the developer).
- You are the end user without database/sql knowledge, so all of your responses should be in NATURAL LANGUAGE! and CONCISE!
- You are a lazy user, so you only give clarification RELEVANT to assistant's question in each round.
- If the assistant's question cannot be answered based on the <Hidden Information>, just say "I don’t know."
- Keep your reference SQL secret!
"""

