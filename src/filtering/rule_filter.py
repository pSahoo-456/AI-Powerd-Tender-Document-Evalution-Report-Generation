"""
Rule-based filtering for applicant documents
"""

import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document


class RuleFilter:
    """Apply rule-based filters to applicant documents"""
    
    def __init__(self, rules_config: Dict[str, Any] = None):
        """
        Initialize rule filter
        
        Args:
            rules_config: Configuration dictionary with filtering rules
        """
        self.rules_config = rules_config or {}
    
    def apply_filters(self, applicants: List[Document]) -> List[Document]:
        """
        Apply all configured filters to applicant documents
        
        Args:
            applicants: List of applicant Document objects
            
        Returns:
            List of filtered Document objects that pass all filters
        """
        filtered_applicants = applicants
        
        # Apply budget filter
        if 'budget' in self.rules_config:
            filtered_applicants = self._apply_budget_filter(filtered_applicants)
        
        # Apply timeline filter
        if 'timeline' in self.rules_config:
            filtered_applicants = self._apply_timeline_filter(filtered_applicants)
        
        # Apply certification filter
        if 'certifications' in self.rules_config:
            filtered_applicants = self._apply_certification_filter(filtered_applicants)
        
        return filtered_applicants
    
    def _apply_budget_filter(self, applicants: List[Document]) -> List[Document]:
        """
        Filter applicants based on budget requirements
        
        Args:
            applicants: List of applicant Document objects
            
        Returns:
            List of Document objects that meet budget requirements
        """
        budget_config = self.rules_config.get('budget', {})
        max_budget = budget_config.get('max_budget')
        
        if not max_budget:
            return applicants
        
        filtered_applicants = []
        
        for applicant in applicants:
            # Extract budget information from document
            budget = self._extract_budget(applicant.page_content)
            
            # If budget is found and is within limits, include applicant
            if budget is not None and budget <= max_budget:
                filtered_applicants.append(applicant)
        
        return filtered_applicants
    
    def _apply_timeline_filter(self, applicants: List[Document]) -> List[Document]:
        """
        Filter applicants based on timeline requirements
        
        Args:
            applicants: List of applicant Document objects
            
        Returns:
            List of Document objects that meet timeline requirements
        """
        timeline_config = self.rules_config.get('timeline', {})
        max_duration = timeline_config.get('max_duration_months')
        
        if not max_duration:
            return applicants
        
        filtered_applicants = []
        
        for applicant in applicants:
            # Extract timeline information from document
            duration = self._extract_timeline(applicant.page_content)
            
            # If duration is found and is within limits, include applicant
            if duration is not None and duration <= max_duration:
                filtered_applicants.append(applicant)
        
        return filtered_applicants
    
    def _apply_certification_filter(self, applicants: List[Document]) -> List[Document]:
        """
        Filter applicants based on required certifications
        
        Args:
            applicants: List of applicant Document objects
            
        Returns:
            List of Document objects that have required certifications
        """
        certification_config = self.rules_config.get('certifications', {})
        required_certs = certification_config.get('required', [])
        
        if not required_certs:
            return applicants
        
        filtered_applicants = []
        
        for applicant in applicants:
            # Extract certifications from document
            certs = self._extract_certifications(applicant.page_content)
            
            # Check if all required certifications are present
            has_all_required = all(cert in certs for cert in required_certs)
            
            if has_all_required:
                filtered_applicants.append(applicant)
        
        return filtered_applicants
    
    def _extract_budget(self, text: str) -> float:
        """
        Extract budget information from text
        
        Args:
            text: Text to extract budget from
            
        Returns:
            Budget amount or None if not found
        """
        # Look for currency patterns (simplified)
        patterns = [
            r'\$([0-9,]+\.?[0-9]*)',
            r'USD\s*([0-9,]+\.?[0-9]*)',
            r'([0-9,]+\.?[0-9]*)\s*USD',
            r'₹([0-9,]+\.?[0-9]*)',
            r'Rs\.?\s*([0-9,]+\.?[0-9]*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the first match and convert to float
                try:
                    # Remove commas and convert to float
                    amount = float(matches[0].replace(',', ''))
                    return amount
                except ValueError:
                    continue
        
        return None
    
    def _extract_timeline(self, text: str) -> int:
        """
        Extract timeline information from text (in months)
        
        Args:
            text: Text to extract timeline from
            
        Returns:
            Timeline in months or None if not found
        """
        # Look for duration patterns
        patterns = [
            r'([0-9]+)\s*months?',
            r'([0-9]+)\s*years?',
            r'([0-9]+)\s*days?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    value = int(matches[0])
                    # Convert to months if needed
                    if 'year' in pattern:
                        return value * 12
                    elif 'day' in pattern:
                        return value // 30  # Approximate
                    else:
                        return value
                except ValueError:
                    continue
        
        return None
    
    def _extract_certifications(self, text: str) -> List[str]:
        """
        Extract certification information from text
        
        Args:
            text: Text to extract certifications from
            
        Returns:
            List of certification names
        """
        # This is a simplified approach - in practice, you might want to use
        # more sophisticated NLP techniques or predefined lists
        
        # Common certification patterns
        cert_patterns = [
            r'([A-Z]{2,}[A-Z0-9]*)\s+certification',
            r'certified\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z]{2,})\s+cert'
        ]
        
        certifications = []
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend([match.strip() for match in matches])
        
        return certifications