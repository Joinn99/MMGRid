
prompt_template = """You are a recommendation system. Based on the user's purchase/interaction history shown below (arranged chronologically from oldest to newest), predict a new {index} of the item that the user is most likely to interact with next.

User's historical interactive items:
"""

item_template = {
    "title": """Title: {title}
Description: {description}""",
    "sem_id": """Item Index: {sem_id}"""
}

prediction_template = {
    "title": """Recommended Item Title: {title}""",
    "sem_id": """Recommended Item Index: {sem_id}"""
}

t2i_prompt = """Given the following item title and desctription, please output the corresponding Item Index.
Examples:
{examples}
"""

i2t_prompt = """Given the following item index, please output the corresponding item title and desctription.
Examples:
{examples}
"""

text_template = """Title: {title}
Description: {description}"""

index_template = """Item Index: {index}"""