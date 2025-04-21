from openai import OpenAI

class QueryRewriter:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
    
    def rewrite_query(self, query: str, history: list, category: str = None) -> str:
        try:
            print(f"\nQuery Rewrite - Original: '{query}'")
            
            # If no history, only expand abbreviations
            if not history:
                print("Query Rewrite - No history available, only expanding abbreviations")
                messages = [
                    {"role": "system", "content": "You are an expert ophthalmologist. Expand ophthalmology-specific abbreviations to their complete medical terms."},
                    {"role": "user", "content": f"""
Important Context:
- Examples of ophthalmology-specific expansions:
  * CTR → capsular tension ring (not click-through rate)
  * IOL → intraocular lens
  * EDOF → extended depth of focus
  * VA → visual acuity
  * IOP → intraocular pressure

Rules:
- ONLY expand ophthalmology-related abbreviations to their full medical terms
- Do NOT change anything else in the question
- Do NOT add any additional context

Question: {query}

Expanded question (only expand abbreviations):"""}
                ]
            else:
                # With history, do minimal rewriting
                print(f"Query Rewrite - Using history context for minimal rewrite")
                formatted_history = "\n".join([
                    f"User: {msg['content'] if msg['role'] == 'user' else ''}\nAssistant: {msg['content'] if msg['role'] == 'assistant' else ''}"
                    for msg in history[-2:]
                ])
                
                # Adjust system message and rules based on category
                if category == "iols":
                    system_message = "You are an expert ophthalmologist specializing in the Precizon Presbyopic NVA IOL."
                    rules = """Rules:
- Replace "this lens", "this IOL", "the lens", "the IOL" with "the Precizon Presbyopic NVA IOL"
- Replace pronouns with terms from the immediate last exchange
- Expand ophthalmology abbreviations
- Keep the rewrite minimal and focused
- Do NOT add any medical context or assumptions
- Do NOT elaborate beyond the original question's scope"""
                else:
                    system_message = "You are an expert ophthalmologist."
                    rules = """Rules:
- Replace pronouns ONLY with terms from the immediate last exchange
- Expand ONLY ophthalmology abbreviations
- Keep the rewrite minimal and focused
- Do NOT add any medical context or assumptions
- Do NOT elaborate beyond the original question's scope"""
                
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"""
Last exchange:
{formatted_history}

{rules}

Question: {query}

Rewritten question (minimal changes only):"""}
                ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            
            if rewritten_query and rewritten_query != query:
                print(f"Query Rewrite - Modified: '{rewritten_query}'")
                print("Query Rewrite - Changes made:")
                print(f"  - Original: '{query}'")
                print(f"  - Modified: '{rewritten_query}'")
                return rewritten_query
            
            print("Query Rewrite - No changes needed")
            return query
            
        except Exception as e:
            print(f"Query Rewrite - Error: {str(e)}")
            return query 