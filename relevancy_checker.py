from openai import OpenAI

class RelevancyChecker:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    def is_ophthalmology_related(self, question: str, answer: str, category: str = None) -> tuple[bool, str]:
        """
        Check if the question-answer pair is related to ophthalmology using GPT-4o.
        For IOL and CTR categories, allows both specific product and general ophthalmology concepts.
        Returns (is_relevant, explanation if not relevant)
        """
        try:
            # Log what's being checked
            print(f"\n🔍 Relevancy Check:")
            print(f"📝 Question: '{question[:100]}...'")
            if answer:
                print(f"📝 Answer: '{answer[:100]}...'")
            
            # Adjust prompt based on whether we're checking just the question or both
            if answer:
                base_prompt = f"""As an ophthalmology expert model, analyze if this question and answer pair is strictly related to ophthalmology or OPHTEC products.

Question: {question}
Answer: {answer}

Analyze both the question and answer to ensure they are focused on:
"""
            else:
                base_prompt = f"""As an ophthalmology expert model, analyze if this question is strictly related to ophthalmology or OPHTEC products.

Question: {question}

Analyze the question to ensure it is focused on:
"""

            base_prompt += """1. Ophthalmology topics, OR
2. OPHTEC products and services

"""
            if category == "iols":
                prompt = base_prompt + """For IOL-related content, accept:
1. Content about the Precizon Presbyopic NVA:
   - Features and specifications including:
     * Continuous Transitional Focus (CTF)
     * Presbyopia correction
     * Transitional focus technology
     * Refractive segments
   - Clinical benefits
   - Surgical techniques
   - Patient outcomes
   - Indications and contraindications
2. General ophthalmology concepts related to:
   - Vision and optics
   - Eye anatomy
   - Surgical procedures
   - Clinical practices
   - Patient care
   - Lens-related topics
   - Presbyopia
   - Multifocal vision
   - Visual acuity at different distances

Rules:
1. Return "YES" if the content is about:
   - Precizon Presbyopic NVA specifically or its features (including CTF)
   - General ophthalmology concepts that are relevant to IOLs
   - Terms and concepts directly related to the product's functionality
2. Return "NO" for:
   - Other IOL models or brands
   - Non-ophthalmology content
3. Provide a clear explanation if not relevant

Format your response exactly as:
RELEVANT: YES/NO
EXPLANATION: [Only if NO, explain why it's not ophthalmology-related]"""

            elif category == "ctr":
                prompt = base_prompt + """For CTR-related content, accept:
1. Content about OPHTEC CTR models:
   - RingJect Model 376
   - RingJect Model 375
   - CTR Model 275 12/10
   - CTR Model 276 13/11
   - Features and specifications
   - Clinical applications
   - Surgical techniques
2. General ophthalmology concepts related to:
   - Capsular support
   - Cataract surgery
   - Eye anatomy
   - Surgical procedures
   - Clinical practices
   - Patient care
   - Lens stability and support

Rules:
1. Return "YES" if the content is about:
   - OPHTEC CTR models specifically
   - General ophthalmology concepts relevant to CTRs and cataract surgery
2. Return "NO" for:
   - Other CTR brands
   - Non-ophthalmology content
3. Provide a clear explanation if not relevant

Format your response exactly as:
RELEVANT: YES/NO
EXPLANATION: [Only if NO, explain why it's not ophthalmology-related]"""
            else:
                prompt = base_prompt + """Specific Topics to Check For:
1. Ophthalmology Topics:
   - Eye anatomy and conditions
   - Ophthalmic procedures
   - Vision and visual system
   - Eye care and treatment
2. OPHTEC Products:
   - IOLs (Intraocular Lenses)
   - CTRs (Capsular Tension Rings)
   - Other OPHTEC devices or services

Rules:
1. Return "YES" only if BOTH conditions are met:
   - Content is ophthalmology-related OR about OPHTEC products
   - Answer provides relevant information to the query
2. Return "NO" for:
   - Non-medical or non-eye related topics
   - Questions about other medical specialties
   - Responses that don't address the query appropriately
3. Provide a clear explanation if not relevant

Format your response exactly as:
RELEVANT: YES/NO
EXPLANATION: [Only if NO, explain why it's not appropriate for this system]"""

            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert ophthalmology model trained to identify ophthalmology-related content with high precision."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4o",
                temperature=0,
                max_tokens=150
            )

            result = response.choices[0].message.content.strip()
            relevant = "RELEVANT: YES" in result
            explanation = result.split("EXPLANATION: ")[1] if "EXPLANATION: " in result else ""
            
            # Enhanced logging of the check result
            print(f"📋 Result: {'Relevant' if relevant else 'Not Relevant'}")
            if explanation:
                print(f"📝 Explanation: {explanation}")
            
            return relevant, explanation

        except Exception as e:
            print(f"Error in relevancy check: {str(e)}")
            return True, ""  # Default to allowing the response if check fails 