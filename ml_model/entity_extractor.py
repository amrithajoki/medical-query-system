"""
Entity Extractor for Medical Imaging Queries
Extracts modality (CT, MRI, etc.) and body_part (chest, brain, etc.) from text
"""

import re
from typing import Dict, Optional


class EntityExtractor:
    """
    Extract medical entities from natural language queries.
    Uses rule-based matching with comprehensive medical vocabulary.
    """
    
    # Comprehensive modality mappings
    MODALITIES = {
        'CT': ['ct', 'computed tomography', 'cat scan'],
        'MRI': ['mri', 'magnetic resonance', 'mr imaging'],
        'X-ray': ['x-ray', 'xray', 'radiograph', 'plain film'],
        'ultrasound': ['ultrasound', 'sonography', 'doppler', 'us', 'echo'],
        'PET': ['pet', 'positron emission', 'pet scan'],
        'mammography': ['mammography', 'mammogram', 'breast imaging'],
        'fluoroscopy': ['fluoroscopy', 'fluoro'],
        'dexa': ['dexa', 'bone density', 'dxa']
    }
    
    # Comprehensive body part mappings
    BODY_PARTS = {
        'head': ['head', 'cranium', 'skull'],
        'brain': ['brain', 'cerebral', 'intracranial'],
        'neck': ['neck', 'cervical soft tissue'],
        'chest': ['chest', 'thorax', 'thoracic'],
        'lung': ['lung', 'pulmonary', 'lungs'],
        'heart': ['heart', 'cardiac', 'coronary'],
        'abdomen': ['abdomen', 'abdominal'],
        'pelvis': ['pelvis', 'pelvic'],
        'spine': ['spine', 'spinal', 'vertebral', 'cervical spine', 'lumbar spine', 'thoracic spine'],
        'shoulder': ['shoulder', 'glenohumeral'],
        'elbow': ['elbow'],
        'wrist': ['wrist', 'carpal'],
        'hand': ['hand', 'fingers', 'thumb'],
        'hip': ['hip', 'femoral head'],
        'knee': ['knee', 'patella'],
        'ankle': ['ankle'],
        'foot': ['foot', 'toes'],
        'breast': ['breast', 'mammary'],
        'liver': ['liver', 'hepatic'],
        'kidney': ['kidney', 'renal', 'kidneys'],
        'pancreas': ['pancreas', 'pancreatic'],
        'spleen': ['spleen', 'splenic'],
        'gallbladder': ['gallbladder', 'gb'],
        'bladder': ['bladder', 'urinary bladder'],
        'prostate': ['prostate'],
        'uterus': ['uterus', 'uterine'],
        'ovary': ['ovary', 'ovaries', 'ovarian'],
        'thyroid': ['thyroid'],
        'sinus': ['sinus', 'sinuses', 'paranasal'],
        'orbit': ['orbit', 'orbital', 'eye socket'],
        'temporal bone': ['temporal bone', 'mastoid'],
        'carotid': ['carotid', 'carotid artery'],
        'aorta': ['aorta', 'aortic'],
        'vein': ['vein', 'venous', 'veins'],
        'artery': ['artery', 'arterial', 'arteries']
    }
    
    def extract(self, query: str) -> Dict[str, Optional[str]]:
        """
        Extract modality and body_part from a query string.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary with 'modality' and 'body_part' keys
            
        Example:
            >>> extractor = EntityExtractor()
            >>> extractor.extract("How many CT scans of the chest?")
            {'modality': 'CT', 'body_part': 'chest'}
        """
        query_lower = query.lower()
        
        # Extract modality
        modality = self._extract_modality(query_lower)
        
        # Extract body part
        body_part = self._extract_body_part(query_lower)
        
        return {
            "modality": modality,
            "body_part": body_part
        }
    
    def _extract_modality(self, query_lower: str) -> Optional[str]:
        """Extract imaging modality from query using word boundaries"""
        for canonical_name, variants in self.MODALITIES.items():
            for variant in variants:
                # Use word boundary regex to avoid partial matches
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, query_lower):
                    return canonical_name
        return None
    
    def _extract_body_part(self, query_lower: str) -> Optional[str]:
        """
        Extract body part from query.
        Tries longer phrases first to avoid incorrect matches.
        """
        # Sort by longest variant first (e.g., "temporal bone" before "bone")
        sorted_parts = sorted(
            self.BODY_PARTS.items(),
            key=lambda x: max(len(v) for v in x[1]),
            reverse=True
        )
        
        for canonical_name, variants in sorted_parts:
            for variant in variants:
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, query_lower):
                    return canonical_name
        return None
    
    def get_all_modalities(self) -> list:
        """Return list of all supported modality canonical names"""
        return list(self.MODALITIES.keys())
    
    def get_all_body_parts(self) -> list:
        """Return list of all supported body part canonical names"""
        return list(self.BODY_PARTS.keys())


# Example usage and testing
if __name__ == "__main__":
    extractor = EntityExtractor()
    
    # Test queries
    test_queries = [
        "How many CT scans of the chest?",
        "List brain MRI studies",
        "Show me cardiac imaging",
        "Count all ultrasound of the abdomen",
        "What modalities are available?",
        "List all body parts",
        "How many spine X-rays?",
        "Give me knee MRI studies"
    ]
    
    print("=" * 70)
    print("ENTITY EXTRACTION TESTS")
    print("=" * 70)
    
    for query in test_queries:
        entities = extractor.extract(query)
        print(f"\nQuery: {query}")
        print(f"  → Modality: {entities['modality']}")
        print(f"  → Body Part: {entities['body_part']}")
    
    print("\n" + "=" * 70)
    print("AVAILABLE ENTITIES")
    print("=" * 70)
    print(f"\nModalities ({len(extractor.get_all_modalities())}):")
    print(", ".join(extractor.get_all_modalities()))
    
    print(f"\nBody Parts ({len(extractor.get_all_body_parts())}):")
    print(", ".join(extractor.get_all_body_parts()))