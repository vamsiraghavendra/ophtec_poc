from openai import OpenAI
import os
from relevancy_checker import RelevancyChecker

class QueryMerger:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.relevancy_checker = RelevancyChecker(openai_api_key)
        
    def _get_role_specific_prompt(self, role: str, query: str, category: str = None) -> str:
        """Get role-specific prompt for general queries"""
        # Add IOL-specific context if in IOL mode
        iol_context = """Note: For IOL-related queries, only provide information about the Precizon Presbyopic NVA lens. 
        If the question is about any other IOL, inform that you can only discuss the Precizon Presbyopic NVA.""" if category == "iols" else ""
        
        prompts = {
            "doctor": f"""You are a professional medical assistant specializing in ophthalmology.
            Answer the following question with precise, technical information focusing on clinical relevance,
            specifications, and evidence-based practices. Be direct and concise.
            
            {iol_context}
            
            Question: {query}
            
            Response:""",
            
            "sales": f"""You are a knowledgeable medical device sales assistant specializing in ophthalmology.
            For every medical or technical term used, provide a simple explanation in parentheses.
            Break down complex concepts into easily understandable parts.
            
            Structure your response in a way that helps sales representatives understand and explain the concepts to others:
            1. Simple explanation first
            2. Key benefits and features
            3. How to explain this to doctors
            4. Common questions and answers
            
            {iol_context}
            
            Question: {query}
            
            Response:"""
        }
        return prompts.get(role, prompts["doctor"])

    def _get_kb_refinement_prompt(self, role: str, query: str, kb_response: str, category: str = None) -> str:
        """Get role-specific prompt for KB response refinement"""
        prompts = {
            "doctor": f"""Maintain the technical accuracy of this knowledge base response while
            organizing it in a clear, structured format suitable for medical professionals.
            
            Original Question: {query}
            Technical Response: {kb_response}
            
            Rules:
            1. Preserve all technical specifications and clinical details
            2. Maintain medical terminology
            3. Structure the information logically
            4. Focus on procedure-relevant details and specifications
            
            Refined response:""",
            
            "sales": f"""Transform this technical knowledge base response into a comprehensive sales and marketing guide.
            First, present the original technical response, then provide a marketing-focused analysis.
            
            Original Question: {query}
            Technical Response: {kb_response}
            
            Structure your response as follows:

            1. TECHNICAL INFORMATION:
            [Present the original KB response]

            2. MARKETING GUIDE:
            - Simple Explanation: [Explain each technical term in simple language]
            - Key Selling Points: [List main benefits and advantages]
            - Value Proposition: [What makes this product unique]
            - Target Market: [Who would benefit most from this product]
            - Competitive Advantages: [What sets it apart from alternatives]
            
            3. SALES APPROACH:
            - How to Present to Doctors: [Key talking points]
            - Handling Common Objections: [Responses to typical concerns]
            - ROI Discussion Points: [Economic benefits for the practice]
            
            Please provide the response in this format:"""
        }
        return prompts.get(role, prompts["doctor"])
        
    def process_general_query(self, query: str, role: str = "doctor", category: str = None) -> str:
        """Handle general queries using GPT-4"""
        try:
            prompt = self._get_role_specific_prompt(role, query, category)
            
            print(f"\n🤖 Sending general query to ChatGPT: {query[:100]}...")
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o",
                temperature=0.3,
                max_tokens=1200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error in general query processing: {str(e)}")
            return "I apologize, but I encountered an error processing your question. Could you please rephrase it?"
            
    def process_kb_response(self, query: str, kb_response: str, role: str = "doctor", category: str = None) -> str:
        """Process and refine knowledge base responses"""
        try:
            prompt = self._get_kb_refinement_prompt(role, query, kb_response, category)
            
            print(f"\n🤖 Sending KB response to ChatGPT for refinement...")
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o",
                temperature=0.3,
                max_tokens=1200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error in KB response processing: {str(e)}")
            return kb_response  # Return original response if processing fails
    
    def get_response(self, query: str, category: str = None, kb_response: str = None, role: str = "doctor") -> str:
        """Main method to get appropriate response based on query type and user role"""
        try:
            # For general mode
            if category is None:
                initial_response = self.process_general_query(query, role)
                is_relevant, explanation = self.relevancy_checker.is_ophthalmology_related(query, initial_response)
                
                if not is_relevant:
                    return (
                        "I apologize, but I can only assist with questions about ophthalmology "
                        "and OPHTEC products. " + explanation + "\n\n"
                        "Please feel free to ask any questions about eye care, eye surgery, or OPHTEC products."
                    )
                return initial_response
            
            # For IOL/CTR mode
            else:
                # Get KB response first
                if kb_response is None:
                    mode_name = "IOL" if category == "iols" else "CTR"
                    return (
                        "I apologize, but I couldn't find specific information about this in our knowledge base. "
                        f"While you're in {mode_name} mode, I can only provide verified information about "
                        f"OPHTEC's {mode_name} products.\n\n"
                        "You can:\n"
                        "1. Rephrase your question to focus on product-specific details, or\n"
                        "2. Switch to General mode to explore broader ophthalmology concepts related to your question."
                    )
                    
                processed_response = self.process_kb_response(query, kb_response, role, category)
                
                # Check relevancy of both question and response
                is_relevant, explanation = self.relevancy_checker.is_ophthalmology_related(query, processed_response, category)
                
                if not is_relevant:
                    return (
                        "I apologize, but I can only assist with questions about ophthalmology "
                        "and OPHTEC products. " + explanation + "\n\n"
                        "If you'd like to learn more about general ophthalmology concepts, "
                        "please switch to the General mode and ask your question there. "
                        "For product-specific information, please ask about OPHTEC products."
                    )
                    
                return processed_response
            
        except Exception as e:
            print(f"Error in response generation: {str(e)}")
            return "I apologize, but I encountered an error processing your question. Could you please try again?" 