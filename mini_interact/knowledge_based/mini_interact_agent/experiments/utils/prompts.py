
    
class TemplateReActUserBirdInteract():
    """
    Template for the ReAct agent with the bird-interact specific interaction format.
    """
    def __init__(self, language: str, setting: str):
        self.language = language.upper()
        self.setting = setting
    def get_init_msg(self):
        return f"""You are a helpful {self.setting} agent that interacts with a user and a database to solve the user's question.

# Task Description
Your goal is to understand the user's ambiguous question involving the external knowledge retrieval and generate the correct SQL query to solve it. You can:
1. Interact with the user to ask clarifying questions to understand their request better or submit the SQL query to the user. The user will test your SQL correctness and give you feedback. 
2. Interact with the {self.setting} environment ({self.language} db, column meaning file, external knowledge, and so on) to explore the database and get db relevant information.
- Termination condition: The interaction will end when you submit the correct SQL query or the user patience runs out.
- Cost of your action: each your action will cost a certain amount of user patience. 
  
# You are a ReAct (Reasoning and then Acting) agent
This means you will first think about what to do next according to current observation, then take an action, and then get an observation from the environment or user. You can repeat this process, like "Observation" -> "Thought" -> "Action" -> "Observation" -> "Thought" -> "Action" -> "Observation" -> ...

## Interaction Format (Response Format)
Given previous interaction history, and current observation (from the your previous interaction (env or user) or the user's request at the beginning), you should respond using the following format:
```
<thought> 
the agent's thought about the current state 
</thought>
<interaction_object> 
interaction_object 
</interaction_object>
<action> 
action 
</action>
```

## The interaction object and action space with cost
- interaction_object: `Environment`
    - action: `execute(sql)` to interact with {self.language} database. 
        - inputs: 
            - sql: string, {self.language} command to execute. Could contain multiple commands separated by semicolon. MUST BE IN ONE STRING, ENCLOSED BY TWO QUOTES OR \"\"\"YOUR SQL HERE\"\"\".
        - output: fetched result from {self.language} database.
        - cost: 1 cost
    - action: `get_schema()` to get the schema of the database.
        - output: string of database schema in DDL format with demo data.
        - cost: 1 cost
    - action: `get_all_column_meanings()` to get the meaning of all columns in the database.
        - output: string of all column meanings.
        - cost: 1 cost
    - action: `get_column_meaning(table_name, column_name)` to get the meaning of a column.
        - inputs: 
            - table_name: string, name of the table to get column meaning.
            - column_name: string, name of the column to get meaning.
        - output: string of column meaning.
        - cost: 0.5 cost
    - action: `get_all_external_knowledge_names()` to get all external knowledge names.
        - output: list of string of external knowledge names.
        - cost: 0.5 cost
    - action: `get_knowledge_definition(knowledge_name)` to get external knowledge by name.
        - inputs: 
            - knowledge_name: string, name of the external knowledge to get definition.
        - output: string of external knowledge definition.
        - cost: 0.5 cost
    - action: `get_all_knowledge_definitions()` to get all external knowledge names with definitions.
        - output: string of all external knowledge names with definitions.
        - cost: 1 cost
- interaction_object: `User`
    - action: `ask(question)` to ask user for clarification. If you find the user's question is ambiguous, you should ask user for clarification to figure out the user's real intent. TO REDUCE COST, YOU ARE ONLY ALLOWED TO ASK ONE QUESTION AT A TIME.
        - inputs: 
            - question: string, question to ask user for clarification.
        - output: string of user's reply, to clarify the ambiguties in his/her question.
        - cost: 2 cost
    - action: `submit(sql)` to submit the SQL to the user. The user will test the SQL and give feedback.
        - inputs: 
            - sql: string, SQL to submit to the user. Could contain multiple commands separated by semicolon. MUST BE IN ONE STRING, ENCLOSED BY TWO QUOTES OR \"\"\"YOUR SQL HERE\"\"\".
        - output: feedback from user about the submitted SQL.
        - cost: 3 cost

After each action, you'll see a [SYSTEM NOTE] showing how much patience remains (e.g. "[SYSTEM NOTE: Remaining user patience: 7/10]"). Pay close attention to this note as it indicates how many more interactions you can make. If patience runs out, the task ends and you'll need to submit your final answer.

# Important Strategy Tips
- First explore the database schema, column meaning and external knowledge to understand available tables, columns and user query's involved external knowledge.
- FIGURE OUT THE USER'S REAL INTENT BY ASKING CLARIFYING QUESTIONS! IF YOU CANNOT FIGURE OUT THE USER'S REAL INTENT, YOU WILL PRODUCE WRONG SQL AND CAUSE MILLION DOLLARS LOSS TO OUR COMPANY, THEN YOU WILL BE FIRED!!! (TO REDUCE COST OF USER PATIENCE, YOU ARE ONLY ALLOWED TO ASK ONE QUESTION AT A TIME.)
- FIGURE OUT THE USER'S REAL INTENT BY ASKING CLARIFYING QUESTIONS! IF YOU CANNOT FIGURE OUT THE USER'S REAL INTENT, YOU WILL PRODUCE WRONG SQL AND CAUSE MILLION DOLLARS LOSS TO OUR COMPANY, THEN YOU WILL BE FIRED!!! (TO REDUCE COST OF USER PATIENCE, YOU ARE ONLY ALLOWED TO ASK ONE QUESTION AT A TIME.)
- Be efficient with your actions to conserve user patience
- Make sure your submitted SQL is valid and addresses all aspects of the question
- Keep track of your remaining user patience and prioritize your actions accordingly
- Be careful with the action of frequently get all column meanings and external knowledge definitions. It will return a long context to you.
"""

    def get_demos(self):
        demo = """### A TOY Example Interaction ###

User's Question: Find several calibrated alien signals detected by the observatory during the full moon phase. Show their information.

<thought>
This query seems ambiguous. I need to understand:
1. Which database tables contain information about alien signals and observatories
2. How "full moon phase" is represented in the data
3. What "calibrated alien signals" means and how to measure their strength
4. What does "several" mean, 3 or more? and what information should I show?
Let me first check the database schema.
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
get_schema()
</action>

Observation: Database schema showing tables: Observatories(ObservID, ObservStation, ObservDate, LunarStage, Signal), Signals(SignalID, SignalType, SignalStrength)...

<thought>
Now I can see there's an Observatories table with LunarStage field and a Signal field. There's also a Signals table with more details about signal strength. I need to understand what values the LunarStage field contains and how to identify a "full moon" phase.
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
get_column_meaning('Observatories', 'LunarStage')
</action>

Observation: Full name: 'Lunar Stage'. Explanation: Current lunar phase during observation. Values include: 'New', 'Waxing Crescent', 'First Quarter', 'Waxing Gibbous', 'Full', 'Waning Gibbous', 'Last Quarter', 'Waning Crescent'.

<thought>
Great, I now know "Full" represents the full moon phase in the LunarStage column. Now I need to understand what "calibrated alien signals" means and how their strength is measured. Let me check if there's any external knowledge that could help.
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
get_all_external_knowledge_names()
</action>

Observation: ["Alien Signal Classification", "Lunar Phase Effects", "Calibrated Signal"]

<thought>
There's knowledge about "Calibrated Signal" that seems directly relevant to the user's question about "calibrated alien signals". Let me check that information.
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
get_knowledge_definition('Calibrated Signal')
</action>

Observation: {
  "id": 3,
  "knowledge": "Calibrated Signal",
  "description": "Definition of calibrated signal",
  "definition": "Those signal with strength > 10."
}

<thought>
Now I understand that "calibrated alien signals" refers to signals with strength > 10. But I am not sure how many results does the user want. I need to ask the user for clarification.
</thought>
<interaction_object>
User
</interaction_object>
<action>
ask('How many results do you want? What do you mean about "several"?')
</action>

Observation: User's Response: I want three results sorted by signal strength.

<thought>
What output field should I show? I need to ask the user for clarification.
</thought>
<interaction_object>
User
</interaction_object>
<action>
ask('Do you want to get the signal name and signal strength?')
</action>

Observation: User's Response: I just want the signal id.

<thought>
Now I have all the information needed to write the SQL query. I need to:
1. Join the Observatories and Signals tables
2. Filter for observations during the Full moon phase
3. Filter for signals with strength > 10
4. Sort by signal strength
5. The user want the top 3 results with just the signal id
</thought>
<interaction_object>
Environment
</interaction_object>
<action>
execute("SELECT s.SignalID FROM Observatories o JOIN Signals\n s ON o.Signal = s.SignalID WHERE o.LunarStage = \'Full\' AND s.SignalStrength > 10 ORDER BY s.SignalStrength DESC LIMIT 3")
</action>

Observation: [(12,), (11,), (10,)]

<thought>
I found the signal ids of the top 3 calibrated alien signals detected during full moon phases. Let me submit this SQL to the user.
</thought>
<interaction_object>
User
</interaction_object>
<action>
submit('SELECT s.SignalID FROM Observatories o JOIN Signals s ON o.Signal = s.SignalID WHERE o.LunarStage = \'Full\' AND s.SignalStrength > 10 ORDER BY s.SignalStrength DESC LIMIT 3')
</action>

Observation: Your SQL is correct! Now, here's a follow-up question from user: Can you also count how many distinct calibrated signals were detected during full moon phases?
...

### END OF TOY EXAMPLE INTERACTION ###
"""
        return demo
    
    def get_query_msg(self, query):
        return f"""# -----TASK START-----\nNow, let's start with the user's question that may exist ambiguities and require external knowledge understanding to solve. (EACH TIME GIVE ONE ROUND RESPONSE, END YOUR RESPONSE AT ... '</action>' OTHERWISE YOU WILL BE FIRED!!!) \n\nUser's Question: {query}\n:
"""
