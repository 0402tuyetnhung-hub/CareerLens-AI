import pandas as pd
import re

class SkillExtractor:
    def __init__(self, dictionary_path):
        self.skills_df = pd.read_csv(dictionary_path)
        # Create a mapping of normalized name to exact name
        self.skill_map = dict(zip(self.skills_df['normalized_name'], self.skills_df['skill_name']))
        
    def extract_skills(self, text):
        extracted = []
        text_lower = text.lower()
        
        # Simple string matching for demo purposes
        # A real system might use SpaCy PhraseMatcher or regex word boundaries
        for normalized_skill, original_skill in self.skill_map.items():
            # Use regex to find whole words only
            pattern = r'\b' + re.escape(normalized_skill) + r'\b'
            if re.search(pattern, text_lower):
                extracted.append(original_skill)
                
        return list(set(extracted))

if __name__ == "__main__":
    extractor = SkillExtractor('../../dataset/it_skills.csv')
    test_text = "I have 2 years of experience with Java, Spring Boot and rest api. Also git."
    print("Extracted:", extractor.extract_skills(test_text))
